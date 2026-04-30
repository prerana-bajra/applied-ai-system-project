from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

try:
    from .agentic_workflow import run_profile_specific_tuning
    from .google_ai import generate_ai_recommendation_summary
    from .recommender import load_songs, recommend_songs, score_song
except ImportError:
    from agentic_workflow import run_profile_specific_tuning
    from google_ai import generate_ai_recommendation_summary
    from recommender import load_songs, recommend_songs, score_song


def _demo_profiles() -> Dict[str, Dict]:
    """Return a small set of demo user profiles for the Streamlit demo.

    Profiles include common listening personas and a few edge/conflict
    cases used to demonstrate the system's behavior.
    """
    return {
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
        "Conflict: High Energy + Sad": {
            "favorite_genre": "pop",
            "favorite_mood": "sad",
            "target_energy": 0.90,
            "target_tempo_bpm": 70,
            "target_valence": 0.15,
            "target_danceability": 0.80,
            "target_acousticness": 0.10,
        },
        "Conflict: Acoustic Raver": {
            "favorite_genre": "edm",
            "favorite_mood": "chill",
            "target_energy": 0.95,
            "target_tempo_bpm": 160,
            "target_valence": 0.20,
            "target_danceability": 0.95,
            "target_acousticness": 0.95,
        },
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


@st.cache_data(show_spinner=False)
def _load_catalog(csv_path: str) -> List[Dict]:
    """Load and cache the song catalog from CSV for the Streamlit app."""
    return load_songs(csv_path)


@st.cache_data(show_spinner=False)
def _run_tuning(
    songs: List[Dict],
    tuning_profiles: Dict[str, Dict],
    iterations: int,
    top_k: int,
) -> Tuple[Dict[str, Dict], List[Dict]]:
    """Run profile-specific agentic tuning and cache results for the demo.

    Returns a mapping of profile names to tuned candidates and the list of logs.
    """
    return run_profile_specific_tuning(
        songs=songs,
        user_profiles=tuning_profiles,
        iterations=iterations,
        top_k=top_k,
        log_path="logs/agentic_experiment_log.json",
    )


def _ai_summary(user_prefs: Dict, recommendations: List[tuple[Dict, float, str]], mode_label: str) -> str:
    """Return a short AI-generated summary for the provided recommendations."""
    return generate_ai_recommendation_summary(user_prefs, recommendations, mode_label=mode_label)


def _confidence_lookup(user_prefs: Dict, songs: List[Dict]) -> Dict[int, float]:
    """Compute per-song normalized confidence scores for the active profile.

    Normalizes raw scores produced by `score_song` to the [0, 1] range.
    """
    raw_scores: Dict[int, float] = {}
    for song in songs:
        raw_score, _ = score_song(user_prefs, song)
        raw_scores[int(song["id"])] = raw_score

    if not raw_scores:
        return {}

    min_score = min(raw_scores.values())
    max_score = max(raw_scores.values())
    spread = max(max_score - min_score, 1e-9)

    return {
        song_id: max(0.0, min(1.0, (score - min_score) / spread))
        for song_id, score in raw_scores.items()
    }


def _recommendation_table(user_prefs: Dict, songs: List[Dict], top_k: int) -> pd.DataFrame:
    """Return a `pandas.DataFrame` representing top-k recommendations for UI display.

    Columns include rank, title, artist, feature-based score, confidence, and reasoning.
    """
    confidence_by_song = _confidence_lookup(user_prefs, songs)
    recommendations = recommend_songs(user_prefs, songs, k=top_k)

    rows: List[Dict] = []
    for rank, (song, final_score, explanation) in enumerate(recommendations, start=1):
        confidence = confidence_by_song.get(int(song["id"]), 0.0)
        rows.append(
            {
                "Rank": rank,
                "Title": song.get("title", ""),
                "Artist": song.get("artist", ""),
                "Genre": song.get("genre", ""),
                "Mood": song.get("mood", ""),
                "Score": round(float(final_score), 2),
                "ConfidenceScore": round(confidence, 4),
                "Confidence": f"{confidence * 100:.1f}%",
                "Why": explanation,
            }
        )

    return pd.DataFrame(rows)


def _build_user_profile_controls(selected_profile: Dict) -> Dict:
    """Render sidebar controls to optionally edit a selected profile.

    Returns a possibly-modified profile dict based on user input widgets.
    """
    profile = deepcopy(selected_profile)
    st.sidebar.markdown("### Optional custom edits")

    profile["favorite_genre"] = st.sidebar.text_input("Favorite genre", value=str(profile["favorite_genre"]))
    profile["favorite_mood"] = st.sidebar.text_input("Favorite mood", value=str(profile["favorite_mood"]))
    profile["target_energy"] = st.sidebar.slider("Target energy", 0.0, 1.2, float(profile["target_energy"]), 0.01)
    profile["target_tempo_bpm"] = st.sidebar.slider("Target tempo (BPM)", 40, 220, int(profile["target_tempo_bpm"]))
    profile["target_valence"] = st.sidebar.slider("Target valence", -0.2, 1.0, float(profile["target_valence"]), 0.01)
    profile["target_danceability"] = st.sidebar.slider("Target danceability", 0.0, 1.1, float(profile["target_danceability"]), 0.01)
    profile["target_acousticness"] = st.sidebar.slider("Target acousticness", -0.2, 1.0, float(profile["target_acousticness"]), 0.01)

    return profile


def _diff_summary(before: pd.DataFrame, after: pd.DataFrame) -> Dict[str, str]:
    """Return a compact summary of changes between two recommendation tables.

    The result includes titles added, removed, and the change in average score.
    """
    before_titles = set(before["Title"].tolist()) if not before.empty else set()
    after_titles = set(after["Title"].tolist()) if not after.empty else set()

    added = sorted(after_titles - before_titles)
    removed = sorted(before_titles - after_titles)

    avg_before = float(before["Score"].mean()) if not before.empty else 0.0
    avg_after = float(after["Score"].mean()) if not after.empty else 0.0

    return {
        "added": ", ".join(added) if added else "None",
        "removed": ", ".join(removed) if removed else "None",
        "avg_change": f"{(avg_after - avg_before):+.2f}",
    }


def _confidence_distribution_frame(confidence_lookup: Dict[int, float], label: str) -> pd.DataFrame:
    """Build a DataFrame showing binned confidence distribution for plotting."""
    if not confidence_lookup:
        return pd.DataFrame(columns=["Range", "Count", "System"])

    raw = pd.DataFrame({"Confidence": list(confidence_lookup.values())})
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    raw["Range"] = pd.cut(raw["Confidence"], bins=bins, labels=labels, include_lowest=True)
    grouped = raw.groupby("Range", observed=False).size().reset_index(name="Count")
    grouped["System"] = label
    return grouped


def _rank_shift_frame(before: pd.DataFrame, after: pd.DataFrame) -> pd.DataFrame:
    """Create a table capturing rank shifts between baseline and agentic outputs."""
    before_slice = before[["Title", "Rank", "Score", "ConfidenceScore"]].rename(
        columns={
            "Rank": "BaselineRank",
            "Score": "BaselineScore",
            "ConfidenceScore": "BaselineConfidence",
        }
    )
    after_slice = after[["Title", "Rank", "Score", "ConfidenceScore"]].rename(
        columns={
            "Rank": "AgenticRank",
            "Score": "AgenticScore",
            "ConfidenceScore": "AgenticConfidence",
        }
    )

    merged = before_slice.merge(after_slice, how="outer", on="Title")
    merged["Status"] = "Retained"
    merged.loc[merged["BaselineRank"].isna(), "Status"] = "New in Agentic"
    merged.loc[merged["AgenticRank"].isna(), "Status"] = "Dropped after Agentic"

    retained = merged["Status"] == "Retained"
    merged["RankShift"] = 0.0
    merged.loc[retained, "RankShift"] = merged.loc[retained, "BaselineRank"] - merged.loc[retained, "AgenticRank"]
    merged["RankShiftLabel"] = merged["RankShift"].map(lambda value: f"{value:+.0f}")

    return merged.sort_values(by=["Status", "AgenticRank", "BaselineRank"], na_position="last")


def _baseline_weights_frame() -> pd.DataFrame:
    """Return a DataFrame showing default weights for the balanced scoring mode."""
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
    
    total = sum(base_weights.values())
    rows = [
        {
            "Feature": feature,
            "Weight": weight,
            "Percentage": f"{(weight / total * 100):.1f}%",
        }
        for feature, weight in sorted(base_weights.items(), key=lambda x: x[1], reverse=True)
    ]
    return pd.DataFrame(rows)


def _iteration_summary_frame(run_logs: List[Dict], profile_name: str = "") -> pd.DataFrame:
    """Create a DataFrame summarizing each iteration's best candidate and metrics."""
    if not run_logs:
        return pd.DataFrame()

    profile_logs = [e for e in run_logs if e.get("profile_name") == profile_name] if profile_name else run_logs
    if not profile_logs:
        profile_logs = run_logs

    iterations_dict: Dict[int, List[Dict]] = {}
    for entry in profile_logs:
        iteration = entry.get("iteration", 0)
        if iteration not in iterations_dict:
            iterations_dict[iteration] = []
        iterations_dict[iteration].append(entry)

    rows: List[Dict] = []
    for iteration_num in sorted(iterations_dict.keys()):
        entries = iterations_dict[iteration_num]
        best_entry = max(entries, key=lambda e: float(e.get("metrics", {}).get("objective_score", 0)))
        candidate = best_entry.get("candidate", {})
        metrics = best_entry.get("metrics", {})
        overrides = candidate.get("weight_overrides") or {}
        override_summary = ", ".join(f"{k}={v}" for k, v in overrides.items()) if overrides else "None"

        rows.append({
            "Iteration": iteration_num,
            "Candidate": candidate.get("name", "unknown"),
            "Scoring Mode": candidate.get("scoring_mode", "balanced"),
            "Weight Adjustments": override_summary,
            "Genre Hit Rate": f"{float(metrics.get('genre_hit_rate', 0)) * 100:.1f}%",
            "Mood Hit Rate": f"{float(metrics.get('mood_hit_rate', 0)) * 100:.1f}%",
            "Explanation Rate": f"{float(metrics.get('explanation_rate', 0)) * 100:.1f}%",
            "Avg Top Score": f"{float(metrics.get('avg_top_score', 0)):.2f}",
            "Objective Score": f"{float(metrics.get('objective_score', 0)):.4f}",
        })

    return pd.DataFrame(rows)


def _iteration_detailed_diff(
    run_logs: List[Dict],
    baseline_profile: Dict,
    baseline_df: pd.DataFrame,
    songs: List[Dict],
    top_k: int,
    profile_name: str,
) -> List[Dict]:
    """For each iteration, compute the best candidate's recommendations and diff vs baseline.

    Returns a list of dicts (one per iteration) with recommendations, song-level diffs,
    and weight changes so the UI can show exactly how each iteration diverged from the
    rule-based baseline.
    """
    profile_logs = [e for e in run_logs if e.get("profile_name") == profile_name]
    if not profile_logs:
        profile_logs = run_logs

    iterations_dict: Dict[int, List[Dict]] = {}
    for entry in profile_logs:
        iteration = entry.get("iteration", 0)
        if iteration not in iterations_dict:
            iterations_dict[iteration] = []
        iterations_dict[iteration].append(entry)

    baseline_titles = set(baseline_df["Title"].tolist()) if not baseline_df.empty else set()
    base_weights = {
        "genre": 15.0, "mood": 20.0, "energy": 24.0, "valence": 12.0,
        "danceability": 10.0, "tempo": 10.0, "acousticness": 6.0,
        "popularity": 8.0, "decade": 8.0, "instrumental": 5.0,
        "mood_confidence": 3.0, "mood_tags": 7.0,
    }

    result = []
    for iteration_num in sorted(iterations_dict.keys()):
        entries = iterations_dict[iteration_num]
        best_entry = max(entries, key=lambda e: float(e.get("metrics", {}).get("objective_score", 0)))
        candidate = best_entry.get("candidate", {})
        metrics = best_entry.get("metrics", {})

        iter_profile = deepcopy(baseline_profile)
        iter_profile["scoring_mode"] = candidate.get("scoring_mode", "balanced")
        iter_profile["weight_overrides"] = deepcopy(candidate.get("weight_overrides") or {})

        iter_df = _recommendation_table(iter_profile, songs, top_k)
        iter_titles = set(iter_df["Title"].tolist())

        added = sorted(iter_titles - baseline_titles)
        removed = sorted(baseline_titles - iter_titles)

        weight_overrides = candidate.get("weight_overrides") or {}
        weight_changes = [
            {
                "Feature": feature,
                "Baseline Weight": base_weights.get(feature, 0),
                "Agentic Weight": new_weight,
                "Delta": f"{new_weight - base_weights.get(feature, 0):+.1f}",
            }
            for feature, new_weight in weight_overrides.items()
        ]

        result.append({
            "iteration": iteration_num,
            "candidate_name": candidate.get("name", ""),
            "scoring_mode": candidate.get("scoring_mode", "balanced"),
            "mode_changed": candidate.get("scoring_mode", "balanced") != "balanced",
            "weight_overrides": weight_overrides,
            "weight_changes": weight_changes,
            "metrics": metrics,
            "recommendations": iter_df,
            "added": added,
            "removed": removed,
            "all_candidates": entries,
        })

    return result


def _iteration_metrics_frame(run_logs: List[Dict], profile_name: str = "") -> pd.DataFrame:
    """Create a DataFrame for plotting metric evolution across iterations."""
    if not run_logs:
        return pd.DataFrame()

    profile_logs = [e for e in run_logs if e.get("profile_name") == profile_name] if profile_name else run_logs
    if not profile_logs:
        profile_logs = run_logs

    iterations_dict: Dict[int, List[Dict]] = {}
    for entry in profile_logs:
        iteration = entry.get("iteration", 0)
        if iteration not in iterations_dict:
            iterations_dict[iteration] = []
        iterations_dict[iteration].append(entry)

    rows: List[Dict] = []
    for iteration_num in sorted(iterations_dict.keys()):
        entries = iterations_dict[iteration_num]
        best_entry = max(entries, key=lambda e: float(e.get("metrics", {}).get("objective_score", 0)))
        metrics = best_entry.get("metrics", {})

        rows.append({
            "Iteration": iteration_num,
            "Genre": float(metrics.get("genre_hit_rate", 0)),
            "Mood": float(metrics.get("mood_hit_rate", 0)),
            "Explanation": float(metrics.get("explanation_rate", 0)),
            "Objective": float(metrics.get("objective_score", 0)),
        })

    return pd.DataFrame(rows)


def main() -> None:
    """Launch the Streamlit demo app showing recommender outputs and analytics."""
    st.set_page_config(page_title="BeatBuddy 2.0", layout="wide")
    st.title("BeatBuddy 2.0: Baseline, AI Output, and Agentic Tuning")

    st.markdown(
        "This interactive demo shows **what** was recommended, **how** it was scored, and **why** outputs change "
        "after adding the agentic tuning loop."
    )

    st.info(
        "Preloaded demo inputs include at least 3 profiles: High-Energy Pop, Chill Lofi, and Deep Intense Rock "
        "(plus additional edge and conflict profiles)."
    )

    songs = _load_catalog("data/songs.csv")
    profile_templates = _demo_profiles()

    st.sidebar.header("Demo Controls")
    selected_profile_name = st.sidebar.selectbox("Choose an example input profile", list(profile_templates.keys()), index=0)
    selected_profile = _build_user_profile_controls(profile_templates[selected_profile_name])

    top_k = st.sidebar.slider("Top-K recommendations", min_value=3, max_value=8, value=5)
    tune_iterations = st.sidebar.slider("Agentic tuning iterations", min_value=1, max_value=6, value=3)
    enable_diversity = st.sidebar.checkbox("Enable diversity penalty", value=True)
    output_mode = st.sidebar.radio(
        "Output mode",
        ["rule", "ai", "agentic", "agentic-ai"],
        index=0,
        help="Rule uses the deterministic scorer; AI adds a Google-generated summary; agentic tunes the scorer first.",
    )
    ai_mode_enabled = output_mode in {"ai", "agentic-ai"}
    agentic_enabled = output_mode in {"agentic", "agentic-ai"}

    baseline_profile = deepcopy(selected_profile)
    baseline_profile["scoring_mode"] = "balanced"
    baseline_profile["weight_overrides"] = {}
    baseline_profile["enable_diversity_penalty"] = enable_diversity

    best_candidates: Dict[str, Dict] = {}
    run_logs: List[Dict] = []
    if agentic_enabled:
        with st.spinner("Running agentic tuning for comparison..."):
            best_candidates, run_logs = _run_tuning(
                songs,
                {selected_profile_name: selected_profile},
                tune_iterations,
                top_k,
            )

    tuned_profile = deepcopy(selected_profile)
    if agentic_enabled:
        profile_candidate = best_candidates.get(selected_profile_name, {"scoring_mode": "balanced", "weight_overrides": {}})
        tuned_profile["scoring_mode"] = profile_candidate.get("scoring_mode", "balanced")
        tuned_profile["weight_overrides"] = deepcopy(profile_candidate.get("weight_overrides", {}))
    else:
        tuned_profile["scoring_mode"] = "balanced"
        tuned_profile["weight_overrides"] = {}
    tuned_profile["enable_diversity_penalty"] = enable_diversity

    baseline_confidence_lookup = _confidence_lookup(baseline_profile, songs)
    tuned_confidence_lookup = _confidence_lookup(tuned_profile, songs)

    baseline_df = _recommendation_table(baseline_profile, songs, top_k)
    tuned_df = _recommendation_table(tuned_profile, songs, top_k)
    diff = _diff_summary(baseline_df, tuned_df)
    rank_shift_df = _rank_shift_frame(baseline_df, tuned_df)

    baseline_distribution = _confidence_distribution_frame(baseline_confidence_lookup, "Baseline")
    tuned_distribution = _confidence_distribution_frame(
        tuned_confidence_lookup,
        "Agentic" if agentic_enabled else "Rule-Based",
    )
    confidence_distribution = pd.concat([baseline_distribution, tuned_distribution], ignore_index=True)

    summary_col1, summary_col2, summary_col3 = st.columns(3)
    summary_col1.metric("Output mode", output_mode)
    summary_col2.metric("Agentic scoring mode", tuned_profile["scoring_mode"])
    summary_col3.metric("Tuning evaluations logged", len(run_logs))

    tab_recommendations, tab_baseline_explain, tab_agentic_iterations, tab_analytics = st.tabs(
        ["Recommendations: What/How/Why", "Rule-Based Baseline", "Agentic Iterations", "Analytics: Confidence and Rank Shift"]
    )

    with tab_recommendations:
        st.markdown("### Baseline (Rule-Based) vs Active Output")
        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Baseline output")
            st.caption("Balanced scoring mode, no weight overrides")
            st.dataframe(baseline_df.drop(columns=["ConfidenceScore"]), use_container_width=True)

        with right_col:
            st.subheader("Active output")
            if agentic_enabled:
                st.caption(
                    f"Scoring mode: {tuned_profile['scoring_mode']}; overrides: {tuned_profile['weight_overrides']}"
                )
            else:
                st.caption("Rule-based scoring with no agentic tuning")
            st.dataframe(tuned_df.drop(columns=["ConfidenceScore"]), use_container_width=True)

        st.markdown("### What Changed After Agentic Addition")
        change_col1, change_col2 = st.columns(2)
        with change_col1:
            st.write(f"**New songs in tuned top-{top_k}:** {diff['added']}")
        with change_col2:
            st.write(f"**Songs dropped from baseline top-{top_k}:** {diff['removed']}")

        st.markdown("### Why These Outputs Were Generated")
        st.write(
            "Each recommendation includes a feature-level explanation in the **Why** column. "
            "Confidence is computed by normalizing each song's raw score within the full catalog for the active profile."
        )

        if ai_mode_enabled:
            st.markdown("### AI-Generated Summary")
            ai_profile = tuned_profile if agentic_enabled else baseline_profile
            ai_recommendations = recommend_songs(ai_profile, songs, k=top_k)
            st.write(_ai_summary(ai_profile, ai_recommendations, output_mode))

        with st.expander("Show tuning objective details"):
            if run_logs:
                latest = run_logs[-1]
                st.json(
                    {
                        "latest_candidate": latest.get("candidate", {}),
                        "latest_metrics": latest.get("metrics", {}),
                        "profile_specific_candidates": best_candidates,
                    }
                )
            else:
                st.write("No tuning logs available.")

    with tab_baseline_explain:
        st.markdown("### How Rule-Based Scoring Works")
        st.write(
            "The **balanced** baseline uses fixed feature weights to score each song against your profile. "
            "Every song in the catalog receives a score based on how well it matches your preferences."
        )

        st.markdown("#### Feature Weights (Balanced Mode)")
        weights_df = _baseline_weights_frame()
        st.dataframe(weights_df, use_container_width=True)

        st.markdown("#### Scoring Process")
        st.write(
            """
            For each song, the system computes similarity scores for each feature:
            1. **Genre**: Does it match your favorite genre?
            2. **Mood**: Does it match your favorite mood?
            3. **Energy**: Is it close to your target energy level?
            4. **Tempo (BPM)**: Is it close to your target tempo?
            5. **Valence**: Does it match your target emotional tone?
            6. **Danceability**: Is it as danceable as you like?
            7. **Acousticness**: Does it match your acoustic preference?
            8. **Other features**: Popularity, decade, instrumental ratio, mood confidence, mood tags, etc.
            
            Each similarity score (0 to 1) is multiplied by its weight, then summed to produce the final score.
            
            **Formula**: `Total Score = Σ(Feature Similarity × Feature Weight)`
            """
        )

        st.markdown("#### Applied Baseline Profile")
        st.write("**Current settings for this profile:**")
        profile_display = {
            "Favorite Genre": baseline_profile.get("favorite_genre"),
            "Favorite Mood": baseline_profile.get("favorite_mood"),
            "Target Energy": baseline_profile.get("target_energy"),
            "Target Tempo (BPM)": baseline_profile.get("target_tempo_bpm"),
            "Target Valence": baseline_profile.get("target_valence"),
            "Target Danceability": baseline_profile.get("target_danceability"),
            "Target Acousticness": baseline_profile.get("target_acousticness"),
            "Diversity Penalty Enabled": baseline_profile.get("enable_diversity_penalty", True),
        }
        st.json(profile_display)

        st.markdown("#### Baseline Top-5 Recommendations")
        st.write("These are the recommendations produced by balanced baseline scoring:")
        st.dataframe(baseline_df.drop(columns=["ConfidenceScore"]), use_container_width=True)

    with tab_agentic_iterations:
        if agentic_enabled:
            st.markdown("### How the Agentic Process Differs from Rule-Based Scoring")
            st.write(
                "The rule-based baseline uses fixed weights in **balanced** scoring mode — the same settings for every profile. "
                "The agentic loop tests different scoring modes and weight adjustments, evaluating each candidate against "
                "the active profile's preferences. Each iteration builds on the winner of the previous round."
            )

            if run_logs:
                st.markdown("#### Iteration Summary")
                iter_summary_df = _iteration_summary_frame(run_logs, profile_name=selected_profile_name)
                st.dataframe(iter_summary_df, use_container_width=True)

                st.markdown("#### Metric Evolution Across Iterations")
                metrics_df = _iteration_metrics_frame(run_logs, profile_name=selected_profile_name)
                if not metrics_df.empty:
                    st.write(
                        "Objective Score = 45% genre hit + 40% mood hit + 10% explanation rate + 5% avg score. "
                        "The agentic loop maximizes this composite across iterations."
                    )
                    metrics_plot_data = metrics_df.set_index("Iteration")
                    st.line_chart(metrics_plot_data[["Genre", "Mood", "Explanation", "Objective"]])

                st.markdown("#### Per-Iteration Changes vs Rule-Based Baseline")
                st.write(
                    "Each section below shows what the agentic tuner chose for that iteration, "
                    "which weights it changed, and exactly which songs were added or removed "
                    "compared to the fixed rule-based output."
                )

                iter_details = _iteration_detailed_diff(
                    run_logs, baseline_profile, baseline_df, songs, top_k, selected_profile_name
                )

                for detail in iter_details:
                    obj_score = float(detail["metrics"].get("objective_score", 0))
                    is_last = detail["iteration"] == iter_details[-1]["iteration"]
                    label = (
                        f"Iteration {detail['iteration']}: {detail['candidate_name']} "
                        f"— Objective {obj_score:.4f}"
                        + (" ✓ Best" if is_last else "")
                    )
                    with st.expander(label, expanded=is_last):
                        m_col1, m_col2, m_col3 = st.columns(3)
                        mode = detail["scoring_mode"]
                        with m_col1:
                            delta = "changed from baseline" if detail["mode_changed"] else None
                            st.metric("Scoring Mode", mode, delta=delta)
                        with m_col2:
                            st.metric("Objective Score", f"{obj_score:.4f}")
                        with m_col3:
                            st.metric("Weight Overrides", len(detail["weight_overrides"]))

                        if detail["weight_changes"]:
                            st.markdown("**Weight Adjustments vs Baseline:**")
                            st.dataframe(pd.DataFrame(detail["weight_changes"]), use_container_width=True)
                        else:
                            st.info("No weight overrides — scoring mode change only (or same as baseline).")

                        st.markdown("**Song Changes vs Rule-Based Baseline:**")
                        song_col1, song_col2 = st.columns(2)
                        with song_col1:
                            if detail["added"]:
                                st.success(f"New in agentic top-{top_k}: {', '.join(detail['added'])}")
                            else:
                                st.write("No new songs vs baseline.")
                        with song_col2:
                            if detail["removed"]:
                                st.warning(f"Dropped from baseline top-{top_k}: {', '.join(detail['removed'])}")
                            else:
                                st.write("No songs dropped vs baseline.")

                        st.markdown("**Side-by-Side Recommendations:**")
                        comp_col1, comp_col2 = st.columns(2)
                        with comp_col1:
                            st.caption("Rule-Based Baseline (balanced, no overrides)")
                            st.dataframe(baseline_df.drop(columns=["ConfidenceScore"]), use_container_width=True)
                        with comp_col2:
                            st.caption(f"Iteration {detail['iteration']}: {detail['candidate_name']}")
                            st.dataframe(detail["recommendations"].drop(columns=["ConfidenceScore"]), use_container_width=True)

                        with st.expander(f"All candidates tested in iteration {detail['iteration']}"):
                            all_cands_df = pd.DataFrame([
                                {
                                    "Candidate": e.get("candidate", {}).get("name", ""),
                                    "Scoring Mode": e.get("candidate", {}).get("scoring_mode", ""),
                                    "Objective": f"{float(e.get('metrics', {}).get('objective_score', 0)):.4f}",
                                    "Genre Hit": f"{float(e.get('metrics', {}).get('genre_hit_rate', 0)) * 100:.1f}%",
                                    "Mood Hit": f"{float(e.get('metrics', {}).get('mood_hit_rate', 0)) * 100:.1f}%",
                                }
                                for e in detail["all_candidates"]
                            ]).sort_values("Objective", ascending=False)
                            st.dataframe(all_cands_df, use_container_width=True)

                st.markdown("#### Final Tuned Configuration")
                best_profile_candidate = best_candidates.get(selected_profile_name, {})
                fin_col1, fin_col2 = st.columns(2)
                with fin_col1:
                    st.json({
                        "name": best_profile_candidate.get("name", ""),
                        "scoring_mode": best_profile_candidate.get("scoring_mode", ""),
                        "weight_overrides": best_profile_candidate.get("weight_overrides") or {},
                    })
                with fin_col2:
                    st.success(
                        f"Tuning converged on **{best_profile_candidate.get('scoring_mode', 'balanced')}** mode "
                        f"with {len(best_profile_candidate.get('weight_overrides') or {})} weight adjustment(s)."
                    )
            else:
                st.info("No tuning logs available. Run agentic tuning to populate this section.")
        else:
            st.info("Agentic tuning is not enabled. Select 'agentic' or 'agentic-ai' mode to see iteration details.")

    with tab_analytics:
        st.markdown("### Confidence Distribution Across Full Catalog")
        distribution_cols = st.columns(2)

        baseline_plot = baseline_distribution[["Range", "Count"]].set_index("Range")
        tuned_plot = tuned_distribution[["Range", "Count"]].set_index("Range")

        with distribution_cols[0]:
            st.caption("Baseline confidence distribution")
            st.bar_chart(baseline_plot)

        with distribution_cols[1]:
            st.caption("Agentic confidence distribution")
            st.bar_chart(tuned_plot)

        st.markdown("### Rank Shift in Top Recommendations")
        retained_only = rank_shift_df[rank_shift_df["Status"] == "Retained"]
        if not retained_only.empty:
            rank_shift_plot = retained_only[["Title", "RankShift"]].set_index("Title")
            st.caption("Positive shift means the song moved up after agentic tuning")
            st.bar_chart(rank_shift_plot)
        else:
            st.info("No overlapping songs between baseline and agentic top-k for this profile.")

        st.markdown("### Rank Shift Detail Table")
        st.dataframe(
            rank_shift_df[
                [
                    "Title",
                    "Status",
                    "BaselineRank",
                    "AgenticRank",
                    "RankShiftLabel",
                    "BaselineConfidence",
                    "AgenticConfidence",
                ]
            ],
            use_container_width=True,
        )

        with st.expander("Show combined confidence distribution data"):
            st.dataframe(confidence_distribution, use_container_width=True)


if __name__ == "__main__":
    main()
