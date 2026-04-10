"""Step 4 — Translate segments with a local LLM via mlx-lm.

Uses a sliding window of the last N translated sentences as context so the
model can resolve pronouns, match tone, and handle speaker-specific slang.

Model is loaded once at module level and stays resident across all jobs.
"""
import logging
import re

from mlx_vlm import load, generate

from src import config

log = logging.getLogger(__name__)

# ── Model (loaded once, stays in memory across jobs) ─────────────────────────

log.info(f"translate: loading LLM {config.LLM_MODEL} …")
_llm_model, _llm_processor = load(config.LLM_MODEL)
log.info("translate: LLM ready")

# Sliding-window size: how many preceding segments to pass as context.
_CONTEXT_WINDOW = 5

# Full language names for the system prompt.
_LANG_NAMES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "tr": "Turkish",
    "ru": "Russian",
    "nl": "Dutch",
    "cs": "Czech",
    "zh": "Chinese",
    "ja": "Japanese",
    "hu": "Hungarian",
    "ko": "Korean",
}


def translate(segments: list[dict], source_lang: str, target_lang: str) -> list[dict]:
    """
    Translate all segments from source_lang to target_lang using the local LLM.
    Each segment gets the preceding _CONTEXT_WINDOW segments as context.
    If source_lang == target_lang, copies text verbatim.
    Returns segments with "translated_text" added.
    """
    if source_lang.lower() == target_lang.lower():
        log.info(f"translate: source == target ({source_lang}), copying text verbatim")
        return [{**seg, "translated_text": seg["text"]} for seg in segments]

    lang_name = _LANG_NAMES.get(target_lang.lower(), target_lang)
    log.info(f"translate: {len(segments)} segments, {source_lang} → {lang_name}")

    out = []
    context_window: list[str] = []  # rolling list of recent original texts

    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            out.append({**seg, "translated_text": ""})
            continue

        context_text = " ".join(context_window)
        translated = _translate_segment(text, context_text, lang_name)
        out.append({**seg, "translated_text": translated})

        # Advance window with the original text so context is always in the
        # source language — consistent regardless of translation quality.
        context_window.append(text)
        if len(context_window) > _CONTEXT_WINDOW:
            context_window.pop(0)

        log.debug(f"translate: seg {seg.get('idx', i)} done")

    log.info("translate: done")
    return out


def _strip_think(text: str) -> str:
    """Remove Gemma 4 <|think|>...</|think|> reasoning blocks from output."""
    return re.sub(r'<\|think\|>.*?</\|think\|>', '', text, flags=re.DOTALL).strip()


def _clean(text: str) -> str:
    """Strip whitespace and truncate at the first sign of runaway repetition."""
    text = text.strip()
    text = re.sub(r'(\S+)(\s+\1){5,}.*', r'\1', text)
    return text.strip()


def _translate_segment(target_text: str, context_text: str, target_lang: str) -> str:
    prompt = f"""<start_of_turn>user
You are an expert podcast translator. You will receive the PREVIOUS CONTEXT of a conversation, followed by a TARGET SEGMENT.
1. Think about the gender, tone, slang, and pronouns based on the context.
2. Translate ONLY the TARGET SEGMENT into {target_lang}.

PREVIOUS CONTEXT: {context_text}
TARGET SEGMENT: {target_text}<end_of_turn>
<start_of_turn>model
"""
    raw = generate(
        _llm_model,
        _llm_processor,
        prompt=prompt,
        max_tokens=350,
        verbose=False,
        repetition_penalty=1.2,
    )
    return _clean(_strip_think(raw))
