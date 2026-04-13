"""
Command line runner for the Music Recommender Simulation.

This file helps you quickly run and test your recommender.

You will implement the functions in recommender.py:
- load_songs
- score_song
- recommend_songs
"""

from recommender import load_songs, recommend_songs

# Score(u,i) = 0.55 * SimContent + 0.20 * MoodMatch + 0.15 * GenreMatch
#            + 0.05 * ArtistBonus + 0.05 * Novelty, with each term scaled to [0,1].

def main() -> None:
    csv_path = "data/songs.csv"
    songs = load_songs(csv_path)
    print(f"Loading songs from {csv_path}, ({len(songs)} songs) ...")

    # Starter example taste profile
    user_prefs = {
        "favorite_genre": "pop",
        "favorite_mood": "chill",
        "target_energy": 0.45,
        "target_tempo_bpm": 80,
        "target_valence": 0.60,
        "target_danceability": 0.60,
        "target_acousticness": 0.75,
        "likes_acoustic": True,
    }

    recommendations = recommend_songs(user_prefs, songs, k=5)

    print("\nTop recommendations:\n")
    for rec in recommendations:
        # You decide the structure of each returned item.
        # A common pattern is: (song, score, explanation)
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")
        print()


if __name__ == "__main__":
    main()
