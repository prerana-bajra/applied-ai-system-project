from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import csv


def _normalized_similarity(value: float, target: float, max_distance: float) -> float:
    """Return a similarity score in the range [0, 1]."""
    if max_distance <= 0:
        return 0.0
    distance = abs(value - target)
    return max(0.0, 1.0 - (distance / max_distance))


def _parse_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_int(value, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_tag_set(value) -> set[str]:
    if not value:
        return set()
    if isinstance(value, str):
        parts = value.replace(",", ";").split(";")
        return {part.strip().lower() for part in parts if part.strip()}
    return {str(tag).strip().lower() for tag in value if str(tag).strip()}


def _get_scoring_mode(user_prefs: Dict) -> str:
    mode = str(user_prefs.get("scoring_mode", "balanced")).strip().lower()
    if mode in {"genre-first", "genre_first", "genre"}:
        return "genre-first"
    if mode in {"mood-first", "mood_first", "mood"}:
        return "mood-first"
    if mode in {"energy-focused", "energy_focused", "energy"}:
        return "energy-focused"
    return "balanced"


def _get_mode_weights(mode: str) -> Dict[str, float]:
    base_weights = {
        "genre": 15.0,
        "mood": 20.0,
        "energy": 24.0,
        "valence": 12.0,
        "danceability": 10.0,
        "tempo": 10.0,
        "acousticness": 6.0,
        "popularity": 8.0,
        "decade": 8.0,
        "instrumental": 5.0,
        "mood_confidence": 3.0,
        "mood_tags": 7.0,
    }

    if mode == "genre-first":
        base_weights.update({"genre": 32.0, "mood": 14.0, "energy": 20.0, "popularity": 6.0, "decade": 6.0})
    elif mode == "mood-first":
        base_weights.update({"genre": 12.0, "mood": 34.0, "energy": 18.0, "mood_tags": 10.0, "mood_confidence": 5.0})
    elif mode == "energy-focused":
        base_weights.update({"genre": 12.0, "mood": 16.0, "energy": 38.0, "tempo": 14.0, "danceability": 12.0})

    return base_weights


def _apply_weight_overrides(weights: Dict[str, float], overrides: Dict) -> Dict[str, float]:
    """Return a copy of weights with validated numeric overrides applied."""
    if not overrides:
        return dict(weights)

    updated_weights = dict(weights)
    for key, value in overrides.items():
        if key not in updated_weights:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue

        if numeric_value >= 0:
            updated_weights[key] = numeric_value

    return updated_weights


def _select_with_diversity_penalty(
    user_prefs: Dict,
    scored_items: List[Tuple[Dict, float, str]],
    k: int,
) -> List[Tuple[Dict, float, str]]:
    """Greedy reranking that penalizes repeated artists and genres in top results."""
    if k <= 0:
        return []

    enable_diversity = bool(user_prefs.get("enable_diversity_penalty", True))
    if not enable_diversity:
        return sorted(scored_items, key=lambda item: item[1], reverse=True)[:k]

    artist_penalty = float(user_prefs.get("artist_repeat_penalty", 12.0))
    genre_penalty = float(user_prefs.get("genre_repeat_penalty", 9.0))

    remaining = sorted(scored_items, key=lambda item: item[1], reverse=True)
    selected: List[Tuple[Dict, float, str]] = []
    artist_counts: Dict[str, int] = {}
    genre_counts: Dict[str, int] = {}

    while remaining and len(selected) < k:
        best_idx = 0
        best_adjusted = float("-inf")

        for idx, (song, raw_score, _) in enumerate(remaining):
            artist_key = str(song.get("artist", "")).strip().lower()
            genre_key = str(song.get("genre", "")).strip().lower()
            adjusted_score = raw_score - (artist_penalty * artist_counts.get(artist_key, 0)) - (
                genre_penalty * genre_counts.get(genre_key, 0)
            )

            if adjusted_score > best_adjusted:
                best_adjusted = adjusted_score
                best_idx = idx

        song, raw_score, explanation = remaining.pop(best_idx)
        artist_key = str(song.get("artist", "")).strip().lower()
        genre_key = str(song.get("genre", "")).strip().lower()

        repeated_artist = artist_counts.get(artist_key, 0)
        repeated_genre = genre_counts.get(genre_key, 0)
        artist_counts[artist_key] = repeated_artist + 1
        genre_counts[genre_key] = repeated_genre + 1

        diversity_penalty_value = (artist_penalty * repeated_artist) + (genre_penalty * repeated_genre)
        final_score = raw_score - diversity_penalty_value
        if diversity_penalty_value > 0:
            explanation = f"{explanation}; diversity penalty applied ({diversity_penalty_value:.2f})"

        selected.append((song, final_score, explanation))

    return selected

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
    popularity: float = 50.0
    release_decade: int = 2000
    mood_tags: str = ""
    instrumental_ratio: float = 0.5
    mood_confidence: float = 0.75
    explicit_content: int = 0

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
    target_popularity: float = 55.0
    preferred_release_decade: int = 2000
    preferred_mood_tags: str = ""
    target_instrumental_ratio: float = 0.5
    avoid_explicit: bool = False

class Recommender:
    """
    OOP implementation of the recommendation logic.
    Required by tests/test_recommender.py
    """
    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        user_prefs = asdict(user)
        scored_items: List[Tuple[Dict, float, str]] = []
        song_lookup: Dict[int, Song] = {}

        for song in self.songs:
            song_dict = asdict(song)
            song_lookup[song.id] = song
            score, reasons = score_song(user_prefs, song_dict)
            scored_items.append((song_dict, score, "; ".join(reasons)))

        ranked_items = _select_with_diversity_penalty(user_prefs, scored_items, k)
        return [song_lookup[int(item[0]["id"])] for item in ranked_items]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        _, reasons = score_song(asdict(user), asdict(song))
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
                "popularity": _parse_float(row.get("popularity"), 50.0),
                "release_decade": _parse_int(row.get("release_decade"), 2000),
                "mood_tags": row.get("mood_tags", ""),
                "instrumental_ratio": _parse_float(row.get("instrumental_ratio"), 0.5),
                "mood_confidence": _parse_float(row.get("mood_confidence"), 0.75),
                "explicit_content": _parse_int(row.get("explicit_content"), 0),
            }
            songs.append(song)
            
    return songs

def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """Score a song against user preferences."""
    reasons: List[str] = []
    mode = _get_scoring_mode(user_prefs)
    weights = _get_mode_weights(mode)
    weights = _apply_weight_overrides(weights, user_prefs.get("weight_overrides", {}))

    favorite_genre = str(user_prefs.get("favorite_genre", "")).strip().lower()
    favorite_mood = str(user_prefs.get("favorite_mood", "")).strip().lower()

    score = 0.0

    if str(song.get("genre", "")).strip().lower() == favorite_genre and favorite_genre:
        score += weights["genre"]
        reasons.append("genre matches your favorite genre")

    if str(song.get("mood", "")).strip().lower() == favorite_mood and favorite_mood:
        score += weights["mood"]
        reasons.append("mood matches your favorite mood")

    target_energy = float(user_prefs.get("target_energy", song.get("energy", 0.0)))
    target_tempo_bpm = float(user_prefs.get("target_tempo_bpm", song.get("tempo_bpm", 0.0)))
    target_valence = float(user_prefs.get("target_valence", song.get("valence", 0.0)))
    target_danceability = float(user_prefs.get("target_danceability", song.get("danceability", 0.0)))
    target_acousticness = float(user_prefs.get("target_acousticness", song.get("acousticness", 0.0)))
    target_popularity = float(user_prefs.get("target_popularity", 55.0))
    preferred_release_decade = _parse_int(user_prefs.get("preferred_release_decade"), 2000)
    preferred_mood_tags = _parse_tag_set(user_prefs.get("preferred_mood_tags", ""))
    target_instrumental_ratio = float(user_prefs.get("target_instrumental_ratio", 0.5))
    avoid_explicit = bool(user_prefs.get("avoid_explicit", False))

    energy_similarity = _normalized_similarity(float(song.get("energy", 0.0)), target_energy, 1.0)
    tempo_similarity = _normalized_similarity(float(song.get("tempo_bpm", 0.0)), target_tempo_bpm, 180.0)
    valence_similarity = _normalized_similarity(float(song.get("valence", 0.0)), target_valence, 1.0)
    danceability_similarity = _normalized_similarity(float(song.get("danceability", 0.0)), target_danceability, 1.0)
    acousticness_similarity = _normalized_similarity(float(song.get("acousticness", 0.0)), target_acousticness, 1.0)
    popularity_similarity = _normalized_similarity(float(song.get("popularity", 50.0)), target_popularity, 100.0)
    decade_similarity = _normalized_similarity(float(song.get("release_decade", 2000)), float(preferred_release_decade), 40.0)
    instrumental_similarity = _normalized_similarity(float(song.get("instrumental_ratio", 0.5)), target_instrumental_ratio, 1.0)
    mood_confidence = float(song.get("mood_confidence", 0.75))
    song_mood_tags = _parse_tag_set(song.get("mood_tags", ""))
    mood_tag_overlap = 0.0
    if preferred_mood_tags and song_mood_tags:
        mood_tag_overlap = len(preferred_mood_tags & song_mood_tags) / float(max(len(preferred_mood_tags), len(song_mood_tags)))

    content_score = (
        weights["energy"] * energy_similarity
        + weights["valence"] * valence_similarity
        + weights["danceability"] * danceability_similarity
        + weights["tempo"] * tempo_similarity
        + weights["acousticness"] * acousticness_similarity
        + weights["popularity"] * popularity_similarity
        + weights["decade"] * decade_similarity
        + weights["instrumental"] * instrumental_similarity
        + weights["mood_confidence"] * mood_confidence
        + weights["mood_tags"] * mood_tag_overlap
    )
    score += content_score

    if content_score > 0:
        reasons.append("audio features are close to your target preferences")

    if popularity_similarity > 0:
        reasons.append("popularity is close to your preference")

    if decade_similarity > 0:
        reasons.append("release era matches your taste")

    if instrumental_similarity > 0:
        reasons.append("instrumental balance matches your taste")

    if mood_tag_overlap > 0:
        reasons.append("detailed mood tags overlap with your preference")

    if mood_confidence > 0:
        reasons.append("the mood tag is well supported")

    if avoid_explicit and int(song.get("explicit_content", 0)) > 0:
        score -= 15.0
        reasons.append("explicit content is filtered out")

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

    return _select_with_diversity_penalty(user_prefs, scored_songs, k)
