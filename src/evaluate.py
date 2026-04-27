from __future__ import annotations

import argparse
from copy import deepcopy
from typing import Dict, List, Tuple

try:
    from .agentic_workflow import run_agentic_tuning
    from .recommender import load_songs, recommend_songs, score_song
except ImportError:
    from agentic_workflow import run_agentic_tuning
    from recommender import load_songs, recommend_songs, score_song


def _default_profiles() -> Dict[str, Dict]:
    return {
        "High-Energy Pop": {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.90,
            "target_tempo_bpm": 128,
            "target_valence": 0.85,
            "target_danceability": 0.88,
            "target_acousticness": 0.20,
            "likes_acoustic": False,
        },
        "Chill Lofi": {
            "favorite_genre": "lofi",
            "favorite_mood": "chill",
            "target_energy": 0.30,
            "target_tempo_bpm": 78,
            "target_valence": 0.55,
            "target_danceability": 0.45,
            "target_acousticness": 0.92,
            "likes_acoustic": True,
        },
        "Deep Intense Rock": {
            "favorite_genre": "rock",
            "favorite_mood": "intense",
            "target_energy": 0.95,
            "target_tempo_bpm": 142,
            "target_valence": 0.30,
            "target_danceability": 0.40,
            "target_acousticness": 0.10,
            "likes_acoustic": False,
        },
        "Conflict: High Energy + Sad": {
            "favorite_genre": "pop",
            "favorite_mood": "sad",
            "target_energy": 0.90,
            "target_tempo_bpm": 70,
            "target_valence": 0.15,
            "target_danceability": 0.80,
            "target_acousticness": 0.10,
            "likes_acoustic": False,
        },
        "Conflict: Acoustic Raver": {
            "favorite_genre": "edm",
            "favorite_mood": "chill",
            "target_energy": 0.95,
            "target_tempo_bpm": 160,
            "target_valence": 0.20,
            "target_danceability": 0.95,
            "target_acousticness": 0.95,
            "likes_acoustic": True,
        },
        "Edge: Unknown Labels + Out of Range": {
            "favorite_genre": "glitch-folk",
            "favorite_mood": "melancholic-euphoric",
            "target_energy": 1.20,
            "target_tempo_bpm": 260,
            "target_valence": -0.10,
            "target_danceability": 1.10,
            "target_acousticness": -0.20,
            "likes_acoustic": True,
        },
    }


def _normalized_confidence_lookup(user_prefs: Dict, songs: List[Dict]) -> Dict[int, float]:
    raw_scores: Dict[int, float] = {}
    for song in songs:
        raw, _ = score_song(user_prefs, song)
        raw_scores[int(song["id"])] = raw

    if not raw_scores:
        return {}

    min_score = min(raw_scores.values())
    max_score = max(raw_scores.values())
    spread = max(max_score - min_score, 1e-9)

    return {
        song_id: max(0.0, min(1.0, (score - min_score) / spread))
        for song_id, score in raw_scores.items()
    }


def _apply_candidate_to_profiles(profiles: Dict[str, Dict], candidate: Dict) -> Dict[str, Dict]:
    updated: Dict[str, Dict] = {}
    for name, profile in profiles.items():
        p = deepcopy(profile)
        p["scoring_mode"] = candidate.get("scoring_mode", "balanced")
        p["weight_overrides"] = deepcopy(candidate.get("weight_overrides", {}))
        updated[name] = p
    return updated


def _evaluate_profiles(songs: List[Dict], profiles: Dict[str, Dict], top_k: int) -> Tuple[Dict, List[Dict]]:
    profile_count = 0
    genre_hits = 0
    mood_hits = 0
    confidence_total = 0.0
    per_profile: List[Dict] = []

    for profile_name, profile in profiles.items():
        profile_count += 1
        confidence_lookup = _normalized_confidence_lookup(profile, songs)
        recommendations = recommend_songs(profile, songs, k=top_k)

        if not recommendations:
            per_profile.append(
                {
                    "profile": profile_name,
                    "genre_match": False,
                    "mood_match": False,
                    "avg_confidence": 0.0,
                    "top_title": "N/A",
                }
            )
            continue

        favorite_genre = str(profile.get("favorite_genre", "")).strip().lower()
        favorite_mood = str(profile.get("favorite_mood", "")).strip().lower()

        genre_match = favorite_genre and any(
            str(song.get("genre", "")).strip().lower() == favorite_genre for song, _, _ in recommendations
        )
        mood_match = favorite_mood and any(
            str(song.get("mood", "")).strip().lower() == favorite_mood for song, _, _ in recommendations
        )

        if genre_match:
            genre_hits += 1
        if mood_match:
            mood_hits += 1

        confidences = [confidence_lookup.get(int(song.get("id", -1)), 0.0) for song, _, _ in recommendations]
        avg_confidence = sum(confidences) / max(len(confidences), 1)
        confidence_total += avg_confidence

        top_song_title = str(recommendations[0][0].get("title", ""))
        per_profile.append(
            {
                "profile": profile_name,
                "genre_match": bool(genre_match),
                "mood_match": bool(mood_match),
                "avg_confidence": round(avg_confidence, 4),
                "top_title": top_song_title,
            }
        )

    safe_count = max(profile_count, 1)
    summary = {
        "profile_count": profile_count,
        "genre_hit_rate": round(genre_hits / safe_count, 4),
        "mood_hit_rate": round(mood_hits / safe_count, 4),
        "avg_confidence": round(confidence_total / safe_count, 4),
    }
    return summary, per_profile


def _objective_score(summary: Dict) -> float:
    return round(
        (0.45 * float(summary.get("genre_hit_rate", 0.0)))
        + (0.40 * float(summary.get("mood_hit_rate", 0.0)))
        + (0.15 * float(summary.get("avg_confidence", 0.0))),
        4,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluation harness for Music Recommender.")
    parser.add_argument("--csv-path", type=str, default="data/songs.csv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--agentic", action="store_true", help="Enable agentic tuning before evaluation.")
    parser.add_argument("--tune-iterations", type=int, default=3)
    parser.add_argument("--objective-threshold", type=float, default=0.72)
    parser.add_argument("--genre-hit-threshold", type=float, default=0.70)
    parser.add_argument("--mood-hit-threshold", type=float, default=0.55)
    parser.add_argument("--confidence-threshold", type=float, default=0.65)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    songs = load_songs(args.csv_path)
    profiles = _default_profiles()

    agentic_note = "disabled"
    best_candidate: Dict = {"scoring_mode": "balanced", "weight_overrides": {}}

    if args.agentic:
        best_candidate, _ = run_agentic_tuning(
            songs=songs,
            user_profiles=profiles,
            iterations=args.tune_iterations,
            top_k=args.top_k,
            log_path="logs/agentic_experiment_log.json",
        )
        profiles = _apply_candidate_to_profiles(profiles, best_candidate)
        agentic_note = (
            f"enabled (mode={best_candidate.get('scoring_mode', 'balanced')}, "
            f"overrides={best_candidate.get('weight_overrides', {})})"
        )

    summary, per_profile = _evaluate_profiles(songs, profiles, args.top_k)
    objective = _objective_score(summary)

    checks = {
        "genre_hit_rate": float(summary["genre_hit_rate"]) >= float(args.genre_hit_threshold),
        "mood_hit_rate": float(summary["mood_hit_rate"]) >= float(args.mood_hit_threshold),
        "avg_confidence": float(summary["avg_confidence"]) >= float(args.confidence_threshold),
        "objective_score": objective >= float(args.objective_threshold),
    }
    overall_pass = all(checks.values())

    print("=== Evaluation Harness Summary ===")
    print(f"Songs evaluated: {len(songs)}")
    print(f"Profiles evaluated: {summary['profile_count']}")
    print(f"Agentic tuning: {agentic_note}")
    print("--- Metrics ---")
    print(f"genre_hit_rate: {summary['genre_hit_rate']:.4f}")
    print(f"mood_hit_rate: {summary['mood_hit_rate']:.4f}")
    print(f"avg_confidence: {summary['avg_confidence']:.4f}")
    print(f"objective_score: {objective:.4f}")
    print("--- Threshold Checks ---")
    print(f"genre_hit_rate >= {args.genre_hit_threshold:.2f}: {'PASS' if checks['genre_hit_rate'] else 'FAIL'}")
    print(f"mood_hit_rate >= {args.mood_hit_threshold:.2f}: {'PASS' if checks['mood_hit_rate'] else 'FAIL'}")
    print(f"avg_confidence >= {args.confidence_threshold:.2f}: {'PASS' if checks['avg_confidence'] else 'FAIL'}")
    print(f"objective_score >= {args.objective_threshold:.2f}: {'PASS' if checks['objective_score'] else 'FAIL'}")
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print("--- Per Profile Snapshot ---")

    for row in per_profile:
        print(
            f"{row['profile']}: top='{row['top_title']}', "
            f"genre_match={row['genre_match']}, mood_match={row['mood_match']}, "
            f"avg_confidence={row['avg_confidence']:.4f}"
        )

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
