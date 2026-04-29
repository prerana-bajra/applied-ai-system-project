from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from .recommender import recommend_songs
except ImportError:
    from recommender import recommend_songs


def _candidate_signature(candidate: Dict) -> str:
    """Create a stable signature string for a candidate configuration.

    The signature encodes the scoring mode and a sorted list of weight
    override (name, value) pairs so that equivalent configurations can be
    detected across iterations.
    """
    mode = str(candidate.get("scoring_mode", "balanced"))
    overrides = candidate.get("weight_overrides", {}) or {}
    ordered = sorted((str(key), float(value)) for key, value in overrides.items())
    return f"{mode}:{ordered}"


def _evaluate_candidate(
    songs: List[Dict],
    user_profiles: Dict[str, Dict],
    candidate: Dict,
    top_k: int,
) -> Dict:
    """Evaluate a single candidate across multiple user profiles.

    For each profile the function applies the candidate's scoring
    configuration, computes top-k recommendations, and accumulates metrics
    such as genre/mood hit rates, explanation coverage, and average top
    score. Returns a dictionary of aggregated metrics and an objective
    score used for ranking candidates.
    """
    profile_count = 0
    genre_hits = 0
    mood_hits = 0
    explanation_hits = 0
    top_score_total = 0.0

    for _, profile in user_profiles.items():
        profile_count += 1
        run_profile = deepcopy(profile)
        run_profile["scoring_mode"] = candidate.get("scoring_mode", "balanced")
        run_profile["weight_overrides"] = deepcopy(candidate.get("weight_overrides", {}))

        recommendations = recommend_songs(run_profile, songs, k=top_k)
        if not recommendations:
            continue

        favorite_genre = str(run_profile.get("favorite_genre", "")).strip().lower()
        favorite_mood = str(run_profile.get("favorite_mood", "")).strip().lower()

        if favorite_genre and any(str(song.get("genre", "")).strip().lower() == favorite_genre for song, _, _ in recommendations):
            genre_hits += 1

        if favorite_mood and any(str(song.get("mood", "")).strip().lower() == favorite_mood for song, _, _ in recommendations):
            mood_hits += 1

        if all(str(explanation).strip() for _, _, explanation in recommendations):
            explanation_hits += 1

        top_score_total += float(recommendations[0][1])

    safe_count = max(profile_count, 1)
    genre_hit_rate = genre_hits / safe_count
    mood_hit_rate = mood_hits / safe_count
    explanation_rate = explanation_hits / safe_count
    avg_top_score = top_score_total / safe_count

    # Objective keeps behavior interpretable by prioritizing user-intent matches.
    objective_score = (
        0.45 * genre_hit_rate
        + 0.40 * mood_hit_rate
        + 0.10 * explanation_rate
        + 0.05 * min(avg_top_score / 100.0, 1.0)
    )

    return {
        "profile_count": profile_count,
        "genre_hit_rate": round(genre_hit_rate, 4),
        "mood_hit_rate": round(mood_hit_rate, 4),
        "explanation_rate": round(explanation_rate, 4),
        "avg_top_score": round(avg_top_score, 4),
        "objective_score": round(objective_score, 4),
    }


def _build_adjusted_candidate(best_candidate: Dict, best_metrics: Dict, iteration_index: int) -> Dict:
    adjusted = deepcopy(best_candidate)
    overrides = deepcopy(adjusted.get("weight_overrides", {}))

    def get_weight(key: str, default: float) -> float:
        value = overrides.get(key, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    if float(best_metrics.get("genre_hit_rate", 0.0)) < 0.75:
        overrides["genre"] = get_weight("genre", 15.0) + 3.0

    if float(best_metrics.get("mood_hit_rate", 0.0)) < 0.75:
        overrides["mood"] = get_weight("mood", 20.0) + 3.0

    if float(best_metrics.get("avg_top_score", 0.0)) < 70.0:
        overrides["energy"] = get_weight("energy", 24.0) + 2.0
        overrides["tempo"] = get_weight("tempo", 10.0) + 1.0

    if float(best_metrics.get("explanation_rate", 0.0)) < 1.0:
        overrides["mood_tags"] = get_weight("mood_tags", 7.0) + 1.0

    adjusted["name"] = f"iter-{iteration_index + 1}-adjusted"
    adjusted["weight_overrides"] = overrides
    return adjusted


def _load_log_entries(log_path: Path) -> List[Dict]:
    if not log_path.exists():
        return []

    try:
        with log_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError):
        return []

    return data if isinstance(data, list) else []


def _append_log_entries(log_path: Path, entries: List[Dict]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    all_entries = _load_log_entries(log_path)
    all_entries.extend(entries)

    with log_path.open("w", encoding="utf-8") as file:
        json.dump(all_entries, file, indent=2)


def run_agentic_tuning(
    songs: List[Dict],
    user_profiles: Dict[str, Dict],
    iterations: int = 3,
    top_k: int = 5,
    log_path: str = "logs/agentic_experiment_log.json",
) -> Tuple[Dict, List[Dict]]:
    """Run a plan-act-check-adjust loop to tune scoring configuration."""
    iteration_count = max(1, int(iterations))
    pool: List[Dict] = [
        {"name": "balanced", "scoring_mode": "balanced", "weight_overrides": {}},
        {"name": "genre-first", "scoring_mode": "genre-first", "weight_overrides": {}},
        {"name": "mood-first", "scoring_mode": "mood-first", "weight_overrides": {}},
        {"name": "energy-focused", "scoring_mode": "energy-focused", "weight_overrides": {}},
    ]

    logs: List[Dict] = []
    seen_signatures = {_candidate_signature(candidate) for candidate in pool}
    global_best_candidate = deepcopy(pool[0])
    global_best_metrics = {"objective_score": float("-inf")}

    for iteration_index in range(iteration_count):
        evaluated: List[Tuple[Dict, Dict]] = []

        for candidate in pool:
            metrics = _evaluate_candidate(songs, user_profiles, candidate, top_k)
            evaluated.append((candidate, metrics))

            logs.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "iteration": iteration_index + 1,
                    "candidate": deepcopy(candidate),
                    "metrics": metrics,
                }
            )

        evaluated.sort(key=lambda row: float(row[1].get("objective_score", 0.0)), reverse=True)
        best_candidate, best_metrics = evaluated[0]

        if float(best_metrics.get("objective_score", 0.0)) > float(global_best_metrics.get("objective_score", 0.0)):
            global_best_candidate = deepcopy(best_candidate)
            global_best_metrics = deepcopy(best_metrics)

        adjusted_candidate = _build_adjusted_candidate(best_candidate, best_metrics, iteration_index)
        adjusted_signature = _candidate_signature(adjusted_candidate)

        next_pool = [deepcopy(candidate) for candidate, _ in evaluated[:3]]
        if adjusted_signature not in seen_signatures:
            next_pool.append(adjusted_candidate)
            seen_signatures.add(adjusted_signature)

        pool = next_pool

    _append_log_entries(Path(log_path), logs)
    return global_best_candidate, logs