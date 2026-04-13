from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import csv


def _normalized_similarity(value: float, target: float, max_distance: float) -> float:
    """Return a similarity score in the range [0, 1]."""
    if max_distance <= 0:
        return 0.0
    distance = abs(value - target)
    return max(0.0, 1.0 - (distance / max_distance))

@dataclass
class Song:
    """
    Represents a song and its attributes.
    Required by tests/test_recommender.py
    """
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float

@dataclass
class UserProfile:
    """
    Represents a user's taste preferences.
    Required by tests/test_recommender.py
    """
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        scored_songs = sorted(
            self.songs,
            key=lambda song: score_song(
                {
                    "favorite_genre": user.favorite_genre,
                    "favorite_mood": user.favorite_mood,
                    "target_energy": user.target_energy,
                    "likes_acoustic": user.likes_acoustic,
                },
                asdict(song),
            )[0],
            reverse=True,
        )
        return scored_songs[:k]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        _, reasons = score_song(
            {
                "favorite_genre": user.favorite_genre,
                "favorite_mood": user.favorite_mood,
                "target_energy": user.target_energy,
                "likes_acoustic": user.likes_acoustic,
            },
            asdict(song),
        )
        return "; ".join(reasons)

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file."""
    songs: List[Dict] = []

    with open(csv_path, "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            song = {
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": float(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            }
            songs.append(song)
            
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score a song against user preferences."""
    reasons: List[str] = []

    favorite_genre = str(user_prefs.get("favorite_genre", "")).strip().lower()
    favorite_mood = str(user_prefs.get("favorite_mood", "")).strip().lower()

    score = 0.0

    if str(song.get("genre", "")).strip().lower() == favorite_genre and favorite_genre:
        score += 15.0
        reasons.append("genre matches your favorite genre")

    if str(song.get("mood", "")).strip().lower() == favorite_mood and favorite_mood:
        score += 20.0
        reasons.append("mood matches your favorite mood")

    target_energy = float(user_prefs.get("target_energy", song.get("energy", 0.0)))
    target_tempo_bpm = float(user_prefs.get("target_tempo_bpm", song.get("tempo_bpm", 0.0)))
    target_valence = float(user_prefs.get("target_valence", song.get("valence", 0.0)))
    target_danceability = float(user_prefs.get("target_danceability", song.get("danceability", 0.0)))
    target_acousticness = float(user_prefs.get("target_acousticness", song.get("acousticness", 0.0)))

    energy_similarity = _normalized_similarity(float(song.get("energy", 0.0)), target_energy, 1.0)
    tempo_similarity = _normalized_similarity(float(song.get("tempo_bpm", 0.0)), target_tempo_bpm, 180.0)
    valence_similarity = _normalized_similarity(float(song.get("valence", 0.0)), target_valence, 1.0)
    danceability_similarity = _normalized_similarity(float(song.get("danceability", 0.0)), target_danceability, 1.0)
    acousticness_similarity = _normalized_similarity(float(song.get("acousticness", 0.0)), target_acousticness, 1.0)

    content_score = (
        24.0 * energy_similarity
        + 12.0 * valence_similarity
        + 10.0 * danceability_similarity
        + 10.0 * tempo_similarity
        + 6.0 * acousticness_similarity
    )
    score += content_score

    if content_score > 0:
        reasons.append("audio features are close to your target preferences")

    favorite_artists = user_prefs.get("favorite_artists") or user_prefs.get("listened_artists") or user_prefs.get("heard_artists")
    if favorite_artists:
        if isinstance(favorite_artists, str):
            favorite_artist_set = {favorite_artists.strip().lower()}
        else:
            favorite_artist_set = {str(artist).strip().lower() for artist in favorite_artists}

        if str(song.get("artist", "")).strip().lower() in favorite_artist_set:
            score += 5.0
            reasons.append("artist is one you already like")
            novelty_score = 0.0
        else:
            novelty_score = 5.0
            reasons.append("artist is new to your listening history")
        score += novelty_score

    if not reasons:
        reasons.append("best overall audio fit among the available songs")

    return score, reasons

def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """Return the top-k scored song recommendations."""
    scored_songs: List[Tuple[Dict, float, str]] = []

    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = "; ".join(reasons)
        scored_songs.append((song, score, explanation))

    scored_songs.sort(key=lambda item: item[1], reverse=True)
    return scored_songs[:k]
