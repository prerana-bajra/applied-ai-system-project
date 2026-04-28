from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Dict, List, Sequence


def _read_key_from_env_file() -> str:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return ""

    try:
        content = env_path.read_text(encoding="utf-8")
    except OSError:
        return ""

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if not line.startswith("GOOGLE_API_KEY="):
            continue

        value = line.split("=", 1)[1].strip()
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        return value.strip()

    return ""


def _read_key_from_streamlit_secrets() -> str:
    try:
        import streamlit as st
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        return ""

    try:
        key = st.secrets.get("GOOGLE_API_KEY", "")
    except StreamlitSecretNotFoundError:
        return ""

    return str(key).strip()


def get_google_api_key() -> str:
    env_value = str(os.getenv("GOOGLE_API_KEY", "")).strip()
    if env_value:
        return env_value

    streamlit_secret = _read_key_from_streamlit_secrets()
    if streamlit_secret:
        return streamlit_secret

    return _read_key_from_env_file()


def _format_recommendations(recommendations: Sequence[tuple[Dict, float, str]]) -> str:
    lines: List[str] = []
    for index, (song, score, explanation) in enumerate(recommendations, start=1):
        lines.append(
            f"{index}. {song.get('title', '')} by {song.get('artist', '')} "
            f"(genre={song.get('genre', '')}, mood={song.get('mood', '')}, score={float(score):.2f})"
        )
        if explanation:
            lines.append(f"   Why: {explanation}")
    return "\n".join(lines)


def _fallback_summary(
    recommendations: Sequence[tuple[Dict, float, str]],
    *,
    reason: str,
) -> str:
    if not recommendations:
        return f"Google AI summary is unavailable ({reason})."

    top_song, top_score, top_explanation = recommendations[0]
    summary_lines = [
        f"Google AI summary is unavailable ({reason}).",
        f"Top recommendation: {top_song.get('title', '')} by {top_song.get('artist', '')} (score={float(top_score):.2f}).",
    ]
    if top_explanation:
        summary_lines.append(f"Why it fits: {top_explanation}.")
    return " ".join(summary_lines)


def generate_ai_recommendation_summary(
    user_prefs: Dict,
    recommendations: Sequence[tuple[Dict, float, str]],
    *,
    mode_label: str,
    model_name: str = "gemini-flash-latest",
    api_key: str | None = None,
) -> str:
    key = api_key or get_google_api_key()
    if not key:
        return "Google AI summary is unavailable because GOOGLE_API_KEY is not set."

    try:
        from google import genai
    except ImportError:
        return "Google AI summary is unavailable because the google-genai package is not installed."

    try:
        from google.genai.errors import ClientError as GoogleClientError
    except ImportError:
        GoogleClientError = None

    client = genai.Client(api_key=key)
    prompt = (
        "You are a music recommendation assistant.\n"
        f"Mode: {mode_label}\n"
        f"User preferences: {user_prefs}\n"
        "Ranked recommendations:\n"
        f"{_format_recommendations(recommendations)}\n\n"
        "Write a concise, user-friendly summary that explains the best pick, why it fits, "
        "and any tradeoffs across the top recommendations. Keep it factual and short."
    )

    requested = str(model_name or "").strip() or "gemini-flash-latest"
    candidate_models: List[str] = [requested]
    for fallback_model in ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash"]:
        if fallback_model not in candidate_models:
            candidate_models.append(fallback_model)

    last_client_error_reason = "Gemini client error unknown"
    response = None
    for candidate_model in candidate_models:
        if GoogleClientError is not None:
            try:
                response = client.models.generate_content(model=candidate_model, contents=prompt)
                break
            except GoogleClientError as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code is None:
                    message = str(exc)
                    match = re.search(r"\b(\d{3})\b", message)
                    if match:
                        status_code = int(match.group(1))
                if status_code == 429:
                    last_client_error_reason = f"Gemini quota exceeded for {candidate_model}"
                    continue
                last_client_error_reason = f"Gemini client error {status_code or 'unknown'} for {candidate_model}"
                continue
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                return f"Google AI summary failed: {exc}"
        else:
            try:
                response = client.models.generate_content(model=candidate_model, contents=prompt)
                break
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                return f"Google AI summary failed: {exc}"

    if response is None:
        return _fallback_summary(recommendations, reason=last_client_error_reason)

    text = getattr(response, "text", "")
    if text and str(text).strip():
        return str(text).strip()

    return _fallback_summary(recommendations, reason="no text was returned by the model")