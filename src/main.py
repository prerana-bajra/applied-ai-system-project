"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""
import argparse
from copy import deepcopy

from textwrap import wrap

from tabulate import tabulate
try:
    from .agentic_workflow import run_profile_specific_tuning
    from .google_ai import generate_ai_recommendation_summary
    from .recommender import load_songs, recommend_songs
except ImportError:
    from agentic_workflow import run_profile_specific_tuning
    from google_ai import generate_ai_recommendation_summary
    from recommender import load_songs, recommend_songs

# Score(u,i) = 0.55 * SimContent + 0.20 * MoodMatch + 0.15 * GenreMatch
#            + 0.05 * ArtistBonus + 0.05 * Novelty, with each term scaled to [0,1].


def _print_recommendation_table(recommendations: list[tuple[dict, float, str]]) -> None:
    """Pretty-print a list of recommendation tuples as a table.

    Args:
        recommendations: Sequence of `(song_dict, score, explanation)` tuples.
    """
    headers = ["#", "Title", "Artist", "Genre", "Score", "Reasons"]
    rows: list[list[str]] = []

    for idx, rec in enumerate(recommendations, start=1):
        song, score, explanation = rec
        wrapped_reasons = "\n".join(wrap(explanation, width=64))
        rows.append(
            [
                str(idx),
                str(song.get("title", "")),
                str(song.get("artist", "")),
                str(song.get("genre", "")),
                f"{score:.2f}",
                wrapped_reasons,
            ]
        )

    print(tabulate(rows, headers=headers, tablefmt="grid"))


def _parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments for the runner."""
    parser = argparse.ArgumentParser(description="Run the music recommender with optional agentic tuning.")
    parser.add_argument(
        "--mode",
        type=str,
        default="rule",
        choices=["rule", "ai", "agentic", "agentic-ai"],
        help="Choose rule-based output, AI-generated output, agentic tuning, or both.",
    )
    parser.add_argument(
        "--agentic-tune",
        action="store_true",
        help="Run the plan-act-check-adjust tuning loop before generating recommendations.",
    )
    parser.add_argument(
        "--tune-iterations",
        type=int,
        default=3,
        help="Number of tuning iterations to run when --agentic-tune is enabled.",
    )
    parser.add_argument(
        "--tune-log-path",
        type=str,
        default="logs/agentic_experiment_log.json",
        help="Path to write agentic experiment logs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of recommendations to show for each profile.",
    )
    parser.add_argument(
        "--ai-model",
        type=str,
        default="gemini-flash-latest",
        help="Google AI model name to use when --mode includes AI output.",
    )
    return parser.parse_args()


def _apply_candidate_to_profiles(user_profiles: dict[str, dict], candidate: dict) -> dict[str, dict]:
    """Apply a tuning candidate to every profile in the provided mapping.

    Produces a shallow copy of `user_profiles` where each profile receives
    the candidate's `scoring_mode` and `weight_overrides` fields.

    Args:
        user_profiles: Mapping from profile name to preference dict.
        candidate: A tuning candidate dict produced by the agentic loop.

    Returns:
        A new dict of tuned profiles suitable for passing to the recommender.
    """
    tuned_profiles: dict[str, dict] = {}
    for profile_name, profile in user_profiles.items():
        tuned_profile = deepcopy(profile)
        tuned_profile["scoring_mode"] = candidate.get("scoring_mode", "balanced")
        tuned_profile["weight_overrides"] = deepcopy(candidate.get("weight_overrides", {}))
        tuned_profiles[profile_name] = tuned_profile
    return tuned_profiles


def main() -> None:
    """Entry point for the CLI runner.

    Loads songs, optionally runs agentic tuning, and prints recommendations
    (and an optional AI summary) for each profile defined in the script.
    """
    args = _parse_args()
    effective_mode = args.mode
    if args.agentic_tune and effective_mode == "rule":
        effective_mode = "agentic"

    csv_path = "data/songs.csv"
    songs = load_songs(csv_path)
    print(f"Loading songs from {csv_path}, ({len(songs)} songs) ...")

    user_profiles = {
        "High-Energy Pop": {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.90,
            "target_tempo_bpm": 128,
            "target_valence": 0.85,
            "target_danceability": 0.88,
            "target_acousticness": 0.20,
        },
        "Chill Lofi": {
            "favorite_genre": "lofi",
            "favorite_mood": "chill",
            "target_energy": 0.30,
            "target_tempo_bpm": 78,
            "target_valence": 0.55,
            "target_danceability": 0.45,
            "target_acousticness": 0.92,
        },
        "Deep Intense Rock": {
            "favorite_genre": "rock",
            "favorite_mood": "intense",
            "target_energy": 0.95,
            "target_tempo_bpm": 142,
            "target_valence": 0.30,
            "target_danceability": 0.40,
            "target_acousticness": 0.10,
        },
        # Adversarial profile: asks for very high energy but a sad mood.
        "Conflict: High Energy + Sad": {
            "favorite_genre": "pop",
            "favorite_mood": "sad",
            "target_energy": 0.90,
            "target_tempo_bpm": 70,
            "target_valence": 0.15,
            "target_danceability": 0.80,
            "target_acousticness": 0.10,
        },
        # Adversarial profile: pushes features to opposite extremes.
        "Conflict: Acoustic Raver": {
            "favorite_genre": "edm",
            "favorite_mood": "chill",
            "target_energy": 0.95,
            "target_tempo_bpm": 160,
            "target_valence": 0.20,
            "target_danceability": 0.95,
            "target_acousticness": 0.95,
        },
        # Adversarial profile: unknown labels and out-of-range targets.
        "Edge: Unknown Labels + Out of Range": {
            "favorite_genre": "glitch-folk",
            "favorite_mood": "melancholic-euphoric",
            "target_energy": 1.20,
            "target_tempo_bpm": 260,
            "target_valence": -0.10,
            "target_danceability": 1.10,
            "target_acousticness": -0.20,
        },
    }

    use_agentic = args.agentic_tune or effective_mode in {"agentic", "agentic-ai"}
    use_ai = effective_mode in {"ai", "agentic-ai"}

    tuned_candidates: dict[str, dict] = {}
    if use_agentic:
        tuned_candidates, logs = run_profile_specific_tuning(
            songs=songs,
            user_profiles=user_profiles,
            iterations=args.tune_iterations,
            top_k=args.top_k,
            log_path=args.tune_log_path,
        )

        print("\n=== Agentic Tuning Summary ===")
        print(f"Profiles tuned: {len(tuned_candidates)}")
        print(f"Iterations logged: {len(logs)}")
        for profile_name, best_candidate in tuned_candidates.items():
            print(
                f"- {profile_name}: mode={best_candidate.get('scoring_mode', 'balanced')}, "
                f"overrides={best_candidate.get('weight_overrides', {})}"
            )
        print()

    for profile_name, user_prefs in user_profiles.items():
        if use_agentic:
            profile_candidate = tuned_candidates.get(profile_name, {"scoring_mode": "balanced", "weight_overrides": {}})
            user_prefs = _apply_candidate_to_profiles({profile_name: user_prefs}, profile_candidate)[profile_name]

        recommendations = recommend_songs(user_prefs, songs, k=args.top_k)

        print(f"\n=== {profile_name} ===\n")
        _print_recommendation_table(recommendations)

        if use_ai:
            summary = generate_ai_recommendation_summary(
                user_prefs,
                recommendations,
                mode_label=effective_mode,
                model_name=args.ai_model,
            )
            print("\n--- AI Summary ---")
            print(summary)


if __name__ == "__main__":
    main()
