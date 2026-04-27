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
    from .agentic_workflow import run_agentic_tuning
    from .recommender import load_songs, recommend_songs
except ImportError:
    from agentic_workflow import run_agentic_tuning
    from recommender import load_songs, recommend_songs

# Score(u,i) = 0.55 * SimContent + 0.20 * MoodMatch + 0.15 * GenreMatch
#            + 0.05 * ArtistBonus + 0.05 * Novelty, with each term scaled to [0,1].


def _print_recommendation_table(recommendations: list[tuple[dict, float, str]]) -> None:
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
    parser = argparse.ArgumentParser(description="Run the music recommender with optional agentic tuning.")
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
    return parser.parse_args()


def _apply_candidate_to_profiles(user_profiles: dict[str, dict], candidate: dict) -> dict[str, dict]:
    tuned_profiles: dict[str, dict] = {}
    for profile_name, profile in user_profiles.items():
        tuned_profile = deepcopy(profile)
        tuned_profile["scoring_mode"] = candidate.get("scoring_mode", "balanced")
        tuned_profile["weight_overrides"] = deepcopy(candidate.get("weight_overrides", {}))
        tuned_profiles[profile_name] = tuned_profile
    return tuned_profiles


def main() -> None:
    args = _parse_args()
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
        # Adversarial profile: asks for very high energy but a sad mood.
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
        # Adversarial profile: pushes features to opposite extremes.
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
        # Adversarial profile: unknown labels and out-of-range targets.
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

    if args.agentic_tune:
        best_candidate, logs = run_agentic_tuning(
            songs=songs,
            user_profiles=user_profiles,
            iterations=args.tune_iterations,
            top_k=args.top_k,
            log_path=args.tune_log_path,
        )

        print("\n=== Agentic Tuning Summary ===")
        print(f"Iterations logged: {len(logs)}")
        print(f"Best scoring mode: {best_candidate.get('scoring_mode', 'balanced')}")
        print(f"Best weight overrides: {best_candidate.get('weight_overrides', {})}\n")

        user_profiles = _apply_candidate_to_profiles(user_profiles, best_candidate)

    for profile_name, user_prefs in user_profiles.items():
        recommendations = recommend_songs(user_prefs, songs, k=args.top_k)

        print(f"\n=== {profile_name} ===\n")
        _print_recommendation_table(recommendations)


if __name__ == "__main__":
    main()
