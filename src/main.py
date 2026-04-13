"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from textwrap import wrap

from tabulate import tabulate

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


def main() -> None:
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

    for profile_name, user_prefs in user_profiles.items():
        recommendations = recommend_songs(user_prefs, songs, k=5)

        print(f"\n=== {profile_name} ===\n")
        _print_recommendation_table(recommendations)


if __name__ == "__main__":
    main()
