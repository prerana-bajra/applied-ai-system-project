# Model Card: BeatBuddy 2.0

## 1. Model Name

BeatBuddy 2.0

## 2. Base Project and Task

Base project: Module 1-3 Music Recommender Simulation.

Task: Recommend top-k songs from a small catalog by matching a user profile to song metadata (genre, mood, energy, tempo, valence, danceability, acousticness), then explain why each song was selected.

## 3. Data Used

- Source: [data/songs.csv](data/songs.csv)
- Size: 50 songs
- Key fields: genre, mood, energy, tempo_bpm, valence, danceability, acousticness, plus extended metadata (popularity, release decade, mood tags, instrumental ratio)
- Coverage note: common genres are represented more than niche combinations or rare moods.

## 4. Intended Use and Non-Intended Use

- Intended use: learning/demo environment for recommendation logic, reliability checks, and agentic tuning workflows.
- Non-intended use: production personalization, high-stakes decisions, or fairness-sensitive ranking at scale.

## 5. Strengths

- Transparent and explainable scoring pipeline.
- Works well for clear profiles (for example, high-energy pop or chill lofi).
- Agentic tuning loop improves repeatability by logging candidate configurations and evaluation metrics.

## 6. Limitations and Biases

- Label bias: genre and mood labels are subjective and can encode cultural bias.
- Representation bias: small curated dataset under-represents unusual tastes and nuanced mood combinations.
- Preference overfit risk: strong feature weights can repeatedly favor a narrow style.
- Missing signals: no lyrics, listening session context, language preference, or user feedback loop.

Observed example: songs with strong energy and genre alignment can rank high even when mood alignment is weaker, which may create repetitive recommendation patterns.

## 7. Reliability and Testing Results

### Evaluation process

- Automated tests: `python -m pytest -q`
- Profile-based checks across six personas:
	- High-Energy Pop
	- Chill Lofi
	- Deep Intense Rock
	- Conflict: High Energy + Sad
	- Conflict: Acoustic Raver
	- Edge: Unknown Labels + Out of Range
- Agentic metrics per candidate:
	- genre_hit_rate
	- mood_hit_rate
	- explanation_rate
	- objective_score

### Quantitative results

- 4/4 tests passed.
- Recent agentic run (12 evaluations) best candidate reached:
	- genre_hit_rate = 0.8333
	- mood_hit_rate = 0.6667
	- explanation_rate = 1.0
	- objective_score = 0.7899

### Reliability findings

- Biggest early failures were software reliability issues (import path and missing dependency), not ranking math.
- Iteration logs improved debugging and made tuning decisions auditable.

## 8. AI Collaboration Reflection

I used AI as a coding assistant for brainstorming, drafting, and implementation acceleration, while validating behavior with tests and runtime checks.

- Helpful AI suggestion: add iteration-level experiment logging (candidate config + metrics). This improved reproducibility and made comparisons across runs straightforward.
- Flawed AI suggestion: an import edit introduced an extra absolute import that broke module execution (`python -m src.main`). I detected this during runtime testing and corrected imports to support package execution reliably.

Main lesson: AI assistance is most useful when paired with verification discipline (tests, command-line validation, and code review).

## 9. Safety and Misuse Considerations

Potential misuse includes manipulating rankings toward narrow commercial goals or reinforcing repetitive content loops. Mitigations used or recommended:

- Keep scoring criteria explicit and inspectable.
- Preserve diversity penalties to reduce over-concentration.
- Log every agentic tuning step for auditability.
- Require human review before adopting new scoring configurations.

## 10. Improvement Roadmap

- Add explicit mood mismatch penalties.
- Expand catalog diversity (genres, moods, language, era).
- Introduce user feedback signals and offline validation datasets.
- Add calibration for objective_score thresholds to better reflect recommendation confidence.
