# BeatBuddy 2.0

BeatBuddy 2.0 is a music recommendation app that ranks songs based on how well each track matches a user taste profile. It combines transparent rule-based scoring with an optional agentic workflow that tunes weights per profile for better personalization.

## What It Does

The app loads songs from `data/songs.csv`, scores them against a user profile, reranks the results with a diversity penalty, and returns the top-k recommendations with explanations. In agentic mode, the system runs a plan-act-check-adjust loop separately for each profile, selects the best tuning candidate for that profile, and applies it before ranking songs.

## Project Structure

- `src/recommender.py`: scoring, ranking, and recommendation helpers.
- `src/agentic_workflow.py`: global and profile-specific tuning loops.
- `src/main.py`: CLI entry point.
- `src/streamlit_app.py`: interactive demo.
- `src/evaluate.py`: evaluation harness with predefined profiles.
- `tests/`: automated tests for scoring and tuning.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you use Google AI summaries, set `GOOGLE_API_KEY` in your environment or a local `.env` file.

## Run the App

```bash
python -m src.main
```

### Agentic CLI Mode

This runs profile-specific tuning before generating recommendations:

```bash
python -m src.main --mode agentic-ai --agentic-tune --tune-iterations 3 --top-k 5
```

### Streamlit Demo

```bash
streamlit run src/streamlit_app.py
```

The Streamlit app lets you select a preset profile, edit its preferences, and compare baseline recommendations against profile-specific tuned results.

## Output Modes

- `rule`: deterministic recommendations only.
- `ai`: deterministic recommendations plus a Google AI summary.
- `agentic`: profile-specific tuning first, then tuned recommendations.
- `agentic-ai`: profile-specific tuning first, then the AI summary.

## Evaluation Harness

```bash
python -m src.evaluate
python -m src.evaluate --agentic --tune-iterations 3
```

The evaluation harness runs multiple profiles, compares baseline and tuned results, and reports metrics such as `genre_hit_rate`, `mood_hit_rate`, `avg_confidence`, and `objective_score`.

## Design Notes

- Rule-based scoring stays transparent and easy to explain.
- Agentic tuning now happens per profile, which improves personalization.
- Diversity reranking reduces repeated artists and genres in the final list.
- All tuning runs are logged to `logs/agentic_experiment_log.json`.

## Tests

```bash
python -m pytest -q
```

## Demo Assets

- Demo video: [assets/beat_buddy_2_0.mov](assets/beat_buddy_2_0.mov)
- Streamlit screenshots: stored in `assets/`
