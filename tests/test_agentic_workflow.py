import json

from src.agentic_workflow import run_agentic_tuning


def test_agentic_tuning_returns_candidate_and_logs(tmp_path):
    songs = [
        {
            "id": 1,
            "title": "Pop One",
            "artist": "A",
            "genre": "pop",
            "mood": "happy",
            "energy": 0.85,
            "tempo_bpm": 122,
            "valence": 0.8,
            "danceability": 0.82,
            "acousticness": 0.2,
            "popularity": 70,
            "release_decade": 2010,
            "mood_tags": "bright;uplifting",
            "instrumental_ratio": 0.2,
            "mood_confidence": 0.9,
            "explicit_content": 0,
        },
        {
            "id": 2,
            "title": "Lofi One",
            "artist": "B",
            "genre": "lofi",
            "mood": "chill",
            "energy": 0.35,
            "tempo_bpm": 78,
            "valence": 0.55,
            "danceability": 0.5,
            "acousticness": 0.88,
            "popularity": 52,
            "release_decade": 2020,
            "mood_tags": "calm;focused",
            "instrumental_ratio": 0.85,
            "mood_confidence": 0.9,
            "explicit_content": 0,
        },
    ]

    user_profiles = {
        "Pop": {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8, "likes_acoustic": False},
        "Lofi": {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.3, "likes_acoustic": True},
    }

    log_path = tmp_path / "agentic_log.json"
    best_candidate, logs = run_agentic_tuning(
        songs=songs,
        user_profiles=user_profiles,
        iterations=2,
        top_k=1,
        log_path=str(log_path),
    )

    assert isinstance(best_candidate, dict)
    assert "scoring_mode" in best_candidate
    assert isinstance(logs, list)
    assert len(logs) > 0
    assert log_path.exists()

    with log_path.open("r", encoding="utf-8") as file:
        saved = json.load(file)
    assert isinstance(saved, list)
    assert len(saved) >= len(logs)