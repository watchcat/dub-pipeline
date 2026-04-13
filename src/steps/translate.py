"""Step 4 — Translate segments with Gemini Flash via the Google GenAI SDK.

All segments are sent in a single batch request with full context, which is
much faster than one API call per segment and gives the model global context.
"""
import json
import logging
import time

from google import genai
from google.genai import types

from src import config

log = logging.getLogger(__name__)

_client = genai.Client(api_key=config.GEMINI_API_KEY)

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
    Translate all segments from source_lang to target_lang using Gemini Flash.
    All segments are sent in one batch request.
    If source_lang == target_lang, copies text verbatim.
    Returns segments with "translated_text" added.
    """
    if source_lang.lower() == target_lang.lower():
        log.info(f"translate: source == target ({source_lang}), copying text verbatim")
        return [{**seg, "translated_text": seg["text"]} for seg in segments]

    lang_name = _LANG_NAMES.get(target_lang.lower(), target_lang)
    log.info(f"translate: {len(segments)} segments, {source_lang} → {lang_name} via Gemini")

    # Build input list: only idx + text needed.
    input_segs = [{"idx": seg.get("idx", i), "text": seg.get("text", "").strip()}
                  for i, seg in enumerate(segments)]

    # Split into batches to avoid hitting Gemini output token limit
    translations: dict[int, str] = {}
    for batch_start in range(0, len(input_segs), _BATCH_SIZE):
        batch = input_segs[batch_start:batch_start + _BATCH_SIZE]
        log.info(f"translate: batch {batch_start // _BATCH_SIZE + 1}, segments {batch_start}–{batch_start + len(batch) - 1}")
        translations.update(_translate_batch(batch, lang_name))

    out = []
    for i, seg in enumerate(segments):
        idx = seg.get("idx", i)
        translated = translations.get(idx)
        if translated is None:
            log.warning(f"translate: seg {idx} missing from Gemini response, using empty string")
            translated = ""
        out.append({**seg, "translated_text": translated})

    log.info("translate: done")
    if out:
        log.info(f"translate: sample — original='{out[0].get('text','')[:60]}' → translated='{out[0].get('translated_text','')[:60]}'")
    return out


_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.ARRAY,
    items=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "idx":             types.Schema(type=types.Type.INTEGER),
            "translated_text": types.Schema(type=types.Type.STRING),
        },
        required=["idx", "translated_text"],
    ),
)


_MAX_RETRIES = 3
_RETRY_DELAY = 10  # seconds, doubled on each attempt
_FALLBACK_MODEL = "gemini-2.0-flash"
_BATCH_SIZE = 50  # segments per Gemini call — avoids output token truncation


def _translate_batch(segments: list[dict], target_lang: str) -> dict[int, str]:
    """Send all segments to Gemini in one call. Returns {idx: translated_text}.
    Retries up to _MAX_RETRIES times on transient errors with exponential backoff.
    Falls back to _FALLBACK_MODEL if the primary model fails all retries."""
    prompt = f"""You are an expert podcast translator. Translate each segment into {target_lang}.

- Translate ONLY the "text" field of each segment.
- Preserve tone, speaker style, and idiomatic expressions.

Segments:
{json.dumps(segments, ensure_ascii=False)}"""

    models = [config.GEMINI_MODEL, _FALLBACK_MODEL]
    for model in models:
        last_exc = None
        delay = _RETRY_DELAY
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = _client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=_RESPONSE_SCHEMA,
                    ),
                )
                result = json.loads(response.text)
                out = {}
                for item in result:
                    idx = item["idx"]
                    translated = item.get("translated_text", "")
                    if not translated:
                        log.warning(f"translate: seg {idx} has empty translated_text")
                    out[idx] = translated
                if model != config.GEMINI_MODEL:
                    log.info(f"translate: succeeded with fallback model {model}")
                return out
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    log.warning(f"translate: {model} failed (attempt {attempt}/{_MAX_RETRIES}): {exc} — retrying in {delay}s")
                    time.sleep(delay)
                    delay *= 2
                else:
                    log.error(f"translate: {model} failed after {_MAX_RETRIES} attempts: {exc} — trying fallback")

    raise last_exc
