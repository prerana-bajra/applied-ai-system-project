# BeatBuddy 2.0: Music Recommender with Agentic Tuning

## Title and Summary

**BeatBuddy 2.0** is a music recommendation system that ranks songs based on how well each track matches a user taste profile. The system started as a rule-based recommender and was extended with an agentic workflow that automatically runs experiments, evaluates performance, and tunes scoring weights — separately per user profile. A Streamlit interactive demo lets you compare three approaches side by side: the fixed rule-based scorer, the same scorer augmented with a Google Gemini natural-language summary, and the agentic tuning loop that finds the best configuration for each specific profile. This matters because it demonstrates a practical Applied AI pattern: combine transparent scoring logic with iterative, testable improvement loops and LLM-based output augmentation.

## Original Project (Modules 1-3)

**Original project name:** Music Recommender Simulation (Modules 1-3).

The original version focused on content-based recommendation using explicit user preferences such as genre, mood, and energy. It could load songs from CSV, compute a score per song, and return top-k recommendations with text explanations. The extension in this repo keeps that core capability and adds automatic experiment runs, per-profile logging-driven tuning, and an interactive Streamlit UI with a Google AI summary layer.

## Design and Architecture: How the System Fits Together

```mermaid
flowchart LR
  U[Input\nUser profile + UI/CLI] --> L[Loader\nload songs from CSV]
  L --> A[Agent\nper-profile plan-act-check-adjust]
  A --> R[Recommender\nscore + diversity re-rank]
  R --> O[Output\nTop-K songs + reasons]
  R --> AI[Google AI\nGemini summary]

  A --> E[Evaluator\nobjective metrics per profile]
  E --> A
  E --> G[Logger\nexperiment history JSON]

  S[Streamlit UI\n4-tab interactive demo] --> U
  S --> O
  S --> AI

  T[Automated Tests\npytest] --> E
  H[Human Review\ninspect outputs and logs] --> O
  H --> G
```

### Architecture Overview

The system has three connected layers. The recommendation layer scores songs and produces ranked outputs with feature-level explanations. The agentic layer runs multiple scoring candidates independently per user profile, evaluates each on profile-based metrics, and updates weights across iterations — ensuring each profile converges to its own best configuration rather than a one-size-fits-all global result. The presentation layer (Streamlit UI) surfaces all of this interactively: four tabs let you inspect what was recommended, how the rule-based scorer works, how each agentic iteration changed the output compared to the baseline, and analytics on confidence distribution and rank shifts across the full catalog.

Data flow: input profile and song catalog → scoring and ranking → recommendation output. During agentic runs, the evaluator feeds metrics back into the tuner loop before final output. The Google AI layer receives the structured ranked output and generates a conversational summary. Human review and automated tests both act as quality checkpoints.

## Main Components

- **Loader** (`src/recommender.py`): Reads `data/songs.csv` and normalizes values.
- **Recommender** (`src/recommender.py`): Computes score from genre/mood matches and numeric feature similarity across 12 weighted features. Supports `balanced`, `genre-first`, `mood-first`, and `energy-focused` scoring modes, plus arbitrary weight overrides per run.
- **Agentic Tuner** (`src/agentic_workflow.py`): Runs a per-profile plan-act-evaluate-adjust loop. Each iteration tests a pool of candidate configurations, ranks them by objective score, keeps the top 3, and generates a mutated candidate that both addresses underperforming metrics and always explores a new feature combination in round-robin order — preventing iterations from stalling on the same configuration.
- **Evaluator** (`src/evaluate.py`): Tracks genre hit rate, mood hit rate, explanation coverage, and composite objective score across 6 representative profiles (standard + conflict + edge cases).
- **Experiment Logger** (`src/agentic_workflow.py`): Writes iteration-level logs to `logs/agentic_experiment_log.json`, tagged with `profile_name` for per-profile traceability.
- **Google AI Integration** (`src/google_ai.py`): Calls the Google Gemini API to convert structured ranked recommendations into a conversational summary. Falls back gracefully to the top recommendation plus explanation if the API is unavailable.
- **Streamlit UI** (`src/streamlit_app.py`): Four-tab interactive demo with sidebar profile selector, output mode toggle (`rule` / `ai` / `agentic` / `agentic-ai`), and controls for top-K and tuning iterations. Tabs: Recommendations (baseline vs active side-by-side), Rule-Based Baseline (weight table + scoring formula), Agentic Iterations (per-iteration diffs vs baseline), and Analytics (confidence distribution + rank shift).
- **Human-in-the-loop Check**: Reviews recommendations and logs for plausibility, diversity, and potential bias.

## Scoring Logic

### Rule-Based Scorer

Every song in the catalog is scored by summing per-feature similarity values multiplied by feature weights:

```
Total Score = Σ (Feature Similarity × Feature Weight)
```

Feature similarities are computed independently (genre/mood exact match → 1 or 0; numeric features → 1 − |song_value − target_value|). The 12 features and their **balanced** baseline weights are:

| Feature | Balanced Weight | What it measures |
|---------|----------------|-----------------|
| energy | 24 | Proximity to target energy level |
| mood | 20 | Exact mood label match |
| genre | 15 | Exact genre label match |
| valence | 12 | Emotional positivity |
| danceability | 10 | Rhythmic danceability |
| tempo | 10 | BPM proximity to target |
| popularity | 8 | Social proof signal |
| decade | 8 | Era / decade preference |
| mood_tags | 7 | Tag-level mood similarity |
| acousticness | 6 | Acoustic texture |
| instrumental | 5 | Vocal vs. instrumental preference |
| mood_confidence | 3 | Data quality signal |

The scorer supports four built-in modes that shift which features dominate:

| Mode | Key weight changes |
|------|--------------------|
| `balanced` | Base weights above — no adjustments |
| `genre-first` | genre→32, mood→14, energy→20 |
| `mood-first` | mood→34, genre→12, energy→18, mood_tags→10 |
| `energy-focused` | energy→38, tempo→14, danceability→12, mood→16 |

After scoring, a diversity penalty is optionally applied to reduce repeated artist/genre in top results.

### Agentic Tuning Loop

The agentic layer treats scoring mode and feature weights as tunable parameters and runs a **plan → act → evaluate → adjust** loop independently per user profile:

1. **Plan** — initialize a pool of 4 candidates (one per built-in mode, no weight overrides).
2. **Act** — run the rule-based scorer for each candidate against the profile's song catalog.
3. **Evaluate** — compute per-candidate metrics and an objective score:

```
Objective Score = 0.45 × genre_hit_rate
               + 0.40 × mood_hit_rate
               + 0.10 × explanation_rate
               + 0.05 × min(avg_top_score / 100, 1.0)
```

Genre and mood hit rates (45 % + 40 % = 85 % combined) dominate because they capture explicit user-intent matches. Raw numeric score contributes only 5 %.

4. **Adjust** — keep the top-3 candidates, then build a mutated candidate from the best:
   - If `genre_hit_rate < 0.75`: boost genre weight by +3
   - If `mood_hit_rate < 0.75`: boost mood weight by +3
   - If `avg_top_score < 70`: boost energy by +2 and tempo by +1
   - If `explanation_rate < 1.0`: boost mood_tags by +1
   - **Always** fine-tune one additional feature in round-robin order (energy → valence → danceability → acousticness → tempo → popularity), scaled by iteration index — this guarantees a unique candidate each iteration even when all metrics already exceed their thresholds.

5. Repeat for N iterations; the candidate with the highest objective score across all iterations becomes the final tuned configuration for that profile.

Each profile converges independently, so a chill lofi listener and a high-energy pop listener produce different scoring modes and weight overrides.

## Setup Instructions

1. Clone the repository and move into the project folder.
2. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) Set the Google API key to enable Gemini summaries. Create a `.env` file in the project root or export the variable:

```bash
export GOOGLE_API_KEY=your_key_here
```

5. Launch the Streamlit demo:

```bash
streamlit run src/streamlit_app.py
```

6. Run the standard recommender from the CLI:

```bash
python -m src.main
```

7. Run with agentic tuning and logging:

```bash
python -m src.main --agentic-tune --tune-iterations 3 --top-k 5
```

8. Run tests:

```bash
python -m pytest -q
```

## Sample Interactions

### Example 1: High-Energy Pop profile

**Input:**

```text
favorite_genre=pop, favorite_mood=happy, target_energy=0.90
```

**Output excerpt:**

```text
1) Sunrise City (pop) - score 124.50
2) Rooftop Lights (pop) - score 113.86
3) Gym Hero (pop) - score 83.08
```

### Example 2: Chill Lofi profile

**Input:**

```text
favorite_genre=lofi, favorite_mood=chill, target_energy=0.30
```

**Output excerpt:**

```text
1) Library Rain (lofi) - score 124.06
2) Focus Flow (lofi) - score 114.12
3) Spacewalk Thoughts (ambient) - score 92.79
```

### Example 3: Per-profile agentic tuning run

**Command:**

```bash
python -m src.main --agentic-tune --tune-iterations 3 --top-k 5
```

**Output excerpt:**

```text
Agentic Tuning Summary
Iterations logged: 12
Best scoring mode (High-Energy Pop): genre-first
Best weight overrides (High-Energy Pop): {'mood': 23.0, 'energy': 26.0}
Best scoring mode (Chill Lofi): balanced
Best weight overrides (Chill Lofi): {'valence': 16.0}
```

Each profile receives its own tuned candidate. The tuner runs independently per profile so a chill lofi listener and a high-energy pop listener converge on different configurations.

### Example 4: Streamlit output modes

In the Streamlit UI, selecting different output modes from the sidebar changes what the active output column shows:

| Mode | What it does |
|------|--------------|
| `rule` | Fixed balanced weights, no tuning |
| `ai` | Same fixed weights + Google Gemini conversational summary |
| `agentic` | Per-profile tuned weights applied, no summary |
| `agentic-ai` | Per-profile tuned weights + Google Gemini summary |

The **Agentic Iterations** tab (visible in `agentic` and `agentic-ai` modes) shows — for each iteration — which scoring mode was selected, which feature weights changed versus the baseline, which songs were added to or dropped from the top-K, and a side-by-side comparison of baseline vs that iteration's recommendations.

## Design Decisions and Trade-offs

1. **Rule-based scoring over black-box modeling**
   - Decision: use explicit scoring terms (genre, mood, tempo, valence, etc.) with fixed weights.
   - Benefit: transparent, debuggable, and easy to explain — every score traces back to a feature comparison.
   - Trade-off: less expressive than a fully learned recommender; cannot capture latent taste patterns.

2. **Per-profile agentic tuning over global tuning**
   - Decision: run the plan-act-evaluate-adjust loop independently for each user profile rather than optimizing a single global candidate across all profiles.
   - Benefit: each profile converges to its own best scoring mode and weight configuration, producing genuinely personalized results rather than a one-size-fits-all solution.
   - Trade-off: compute time scales with number of profiles; N profiles means N independent tuning loops.

3. **Round-robin feature exploration in mutation step**
   - Decision: the adjusted candidate always modifies at least one feature weight in round-robin order (energy → valence → danceability → acousticness → tempo → popularity) in addition to any metric-driven adjustments.
   - Benefit: prevents iterations from stalling when all metrics already exceed their thresholds — the agent keeps exploring rather than repeating the same pool of candidates.
   - Trade-off: introduces exploratory noise even when the current configuration is already good; the objective function must be reliable to distinguish genuine improvements from random variation.

4. **Diversity re-ranking**
   - Decision: penalize repeated artist/genre in top results.
   - Benefit: reduces repetitive recommendations.
   - Trade-off: can lower pure match score for stronger diversity.

5. **Google AI as a presentation layer, not a ranking layer**
   - Decision: Gemini is called only after ranking is complete; it receives structured output (songs, scores, explanations) and converts it to natural language.
   - Benefit: the ranking stays fully deterministic and auditable; the LLM cannot alter the recommendation logic or introduce hallucinated rankings.
   - Trade-off: the natural-language summary quality depends on Gemini availability; fallback returns the top recommendation plus its rule-based explanation.

## Guardrails and Quality Control

The system includes multiple guardrails to ensure recommendations remain transparent, diverse, and aligned with user intent — preventing misuse and maintaining quality:

### **Algorithmic Guardrails**

1. **Objective function weighting** — user-intent matches (genre + mood) comprise 85% of the scoring decision; raw numeric scores only 5%. This prevents the system from being hijacked into engagement-maximization or other adversarial objectives that conflict with user wellbeing.

2. **Metric-driven adjustments** — weight mutations only trigger when specific thresholds are missed:
   - `genre_hit_rate < 0.75` → boost genre weight
   - `mood_hit_rate < 0.75` → boost mood weight
   - `avg_top_score < 70` → boost energy and tempo
   - `explanation_rate < 1.0` → boost mood_tags
   
   This prevents over-tuning when performance is already strong.

3. **Round-robin feature exploration** — each iteration always explores a new feature weight (energy → valence → danceability → acousticness → tempo → popularity, cycling), scaled by iteration index. Guarantees novel candidates even when all metrics are already satisfied — prevents iteration stalling and ensures genuine exploration.

4. **Deduplication via `seen_signatures`** — identical candidate configurations are never re-evaluated. Prevents wasted compute and makes the tuning loop deterministic.

5. **Diversity penalty** — repeated artist/genre in top-K results are penalized, reducing narrow content loops.

### **Transparency and Auditability**

6. **Rule-based scoring (not black-box ML)** — all scores trace directly back to explicit feature weights. Every recommendation includes a feature-level explanation showing which genres/moods matched and how numeric features contributed. No hidden ranking logic.

7. **Google AI as presentation layer only** — Gemini receives the final ranked list with explanations and generates a conversational summary. It cannot alter the recommendation order, inject new songs, or produce hallucinated rankings. The deterministic scorer remains the source of truth.

8. **Graceful fallback** — if the Gemini API is unavailable, the system falls back to the top recommendation plus its rule-based explanation rather than degrading silently.

9. **Per-profile tuning audit trail** — every agentic iteration is logged to `logs/agentic_experiment_log.json` with a `profile_name` tag, enabling per-profile traceability. Each log entry includes timestamp, candidate configuration, metrics, and objective score — supporting full reconstruction and audit of tuning decisions.

10. **Explanation rate enforcement** — a metric ensures that 100% of recommendations across a profile include a non-blank explanation. Any candidate failing this check is discarded, maintaining output quality.

### **Human-in-the-Loop Oversight**

11. **Manual output inspection** — recommendations are reviewed for plausibility, diversity, and absence of bias before deployment decisions. The Streamlit UI's per-iteration diff view makes this inspection structural and efficient.

12. **Policy-level checks** — configurations can be validated against organization-level policies (e.g., no engagement-maximizing objectives, mandatory diversity thresholds) before live deployment.

## Reliability and Evaluation: How I Test and Improve the AI

This project includes multiple reliability checks so performance is measured, not assumed. Evaluation happens at three levels: unit testing, metric-driven validation, and human-guided review.

### **Automated Testing**

- **Unit tests**: `python -m pytest -q` currently reports **4 out of 4 tests passed**, covering core recommendation behavior, weight override application, profile-specific tuning, and end-to-end logging.

### **Metric-Based Evaluation (Per Candidate)**

Every candidate configuration is evaluated using a four-component scorecard:

| Metric | Definition | Target |
|--------|-----------|--------|
| `genre_hit_rate` | Fraction of profiles whose favorite genre appears in top-K | 1.0 (100%) |
| `mood_hit_rate` | Fraction of profiles whose favorite mood appears in top-K | 1.0 (100%) |
| `explanation_rate` | Fraction of profiles where all top-K recommendations have non-blank explanations | 1.0 (100%) |
| `avg_top_score` | Average score of the #1 recommendation across all profiles | 70+ |

**Objective Score Formula** — combines these into a decision metric:

```
Objective Score = (0.45 × genre_hit_rate)
                + (0.40 × mood_hit_rate)
                + (0.10 × explanation_rate)
                + (0.05 × min(avg_top_score / 100, 1.0))
```

The 85% weighting on genre + mood captures explicit user intent; only 5% comes from raw numeric scores. This ensures the optimization loop prioritizes user-stated preferences over engagement-maximizing heuristics.

### **Per-Profile Evaluation**

When running `run_profile_specific_tuning`, evaluation is **per-profile**, not aggregated:

- Each profile runs its own tuning loop independently
- Metrics are computed separately for that profile's preferences
- Log entries are tagged with `profile_name` for traceability
- Allows chill lofi listeners and high-energy pop listeners to converge on different configurations

### **Iteration Tracking and Validation**

- **Logging and auditability**: Every tuning step is stored in `logs/agentic_experiment_log.json` with timestamp, iteration number, candidate configuration, and metrics. Log loading is fail-safe — defaults to empty history if JSON is missing or corrupt.
- **Iteration integrity**: The `seen_signatures` deduplication set prevents the same candidate configuration from being evaluated twice, while the round-robin exploration step guarantees each iteration adds at least one genuinely new configuration to test.
- **Comparison across iterations**: The Streamlit UI displays metric evolution charts (genre hit, mood hit, objective score) across iterations, revealing whether tuning is converging upward or stalling.

### **Human Evaluation**

- **Output review**: Recommendation outputs are manually inspected for plausibility, diversity, absence of repetitive artists/genres, and quality of explanations.
- **Structural inspection**: The Streamlit UI's per-iteration diff view makes review efficient — you can see exactly which songs were added or removed at each step compared to the baseline.
- **Bias and fairness check**: Manual review flags potential biases (e.g., over-representation of mainstream genres, under-representation of niche styles) and diversity issues.

### Quantitative Summary

- **4/4 tests passed** after integrating the agentic workflow.
- In a representative per-profile tuning run (3 iterations, 6 profiles), the `genre-first` scoring mode consistently produced the highest objective scores (~0.79) on standard profiles, while conflict and edge-case profiles converged to different modes.
- Explanation coverage (`explanation_rate`) remained at **1.0** across all runs — every recommendation includes a feature-level reason.
- Main failure mode identified and fixed during development: when all profile metrics exceeded their adjustment thresholds in iteration 1, the mutation step produced a no-op candidate whose signature matched an already-seen configuration, causing all subsequent iterations to repeat the same pool. Fixed by adding guaranteed round-robin feature exploration independent of metric thresholds.

## Testing Summary

### What worked

- Core recommendation behavior passed unit tests.
- Agentic workflow ran end-to-end and produced experiment logs.
- Weight override behavior changed scores as expected.
- Per-profile tuning produced different best candidates for different profile types.
- The Streamlit UI correctly surfaces per-iteration diffs and side-by-side comparisons.

### What did not work initially

- Packaging/import mismatch caused `python -m src.main` to fail before import cleanup.
- Local runtime failed when dependencies were missing until environment setup was completed.
- Agentic iterations produced identical results across all rounds when profile metrics were already strong — the mutation step generated a no-op candidate that was filtered by the deduplication check, leaving the pool unchanged. Fixed by ensuring the adjusted candidate always explores a new feature weight regardless of current metric values.

### What I learned

- Reliability improves when imports support both package and script execution paths.
- Agent loops are most useful when metrics are explicit and logged every iteration.
- Adding tests for both output quality and pipeline behavior prevents silent regressions.
- The mutation step in an agentic loop must guarantee novelty — not just react to failures — otherwise well-performing early iterations cause the loop to stall rather than continue exploring.

## Reflection and Ethics

### What are the limitations or biases in this system?

This recommender is still a rules-and-weights system, so it reflects the assumptions built into those weights. Genre and mood labels can be noisy or culturally biased, and strong genre matching can over-recommend mainstream categories while under-representing niche styles. The dataset is small (50 songs) and curated, so performance may not generalize to real user behavior, multilingual catalogs, or rapidly changing tastes. The Google AI summary layer inherits any biases present in the Gemini model — it can generate confident-sounding text even when the underlying recommendations are weak.

### Could this AI be misused, and how would I prevent that?

A recommender can be misused to push narrow content loops, prioritize commercial goals over user well-being, or quietly suppress certain artists. The agentic tuner specifically could be misused if the objective function were replaced with an engagement-maximizing metric that conflicts with user wellbeing. To reduce misuse, I would keep ranking criteria transparent, retain diversity penalties, and add policy checks for harmful optimization objectives. I would also require logging and human review for configuration changes so that aggressive tuning decisions are auditable before deployment. The experiment log with `profile_name` tagging already supports this kind of audit trail.

### What surprised me while testing reliability?

I expected model-like tuning to be the hardest part, but reliability issues were initially caused by software plumbing: import-path and environment dependency errors. After those were fixed, the system became stable quickly, and metric logging made the tuning loop much easier to reason about. A second surprise was that the agentic iteration loop silently stalled when metrics were already good — the agent appeared to be running but was testing the same three candidates each time. This was only visible by comparing iteration logs closely. It reinforced that the iteration mechanism itself needs to be tested, not just the output quality.

### Collaboration with AI during this project

I used AI as a coding copilot for implementation speed and review support, not as an unquestioned authority.

- Helpful suggestion: the AI proposed adding an experiment log for each tuning iteration (candidate config plus metrics). That made the agentic workflow reproducible and significantly improved debugging and comparison across runs.
- Helpful suggestion: the AI identified that the mutation step needed guaranteed novelty — adding the round-robin feature exploration ensured each iteration tests a genuinely new configuration even when all current metrics are already above threshold.
- Flawed suggestion: one AI-generated import change introduced an extra absolute import that broke module execution (`python -m src.main`). I caught this through runtime testing and corrected the import structure so both package and script paths work correctly.
