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

    translations = _translate_batch(input_segs, lang_name)

    out = []
    for i, seg in enumerate(segments):
        idx = seg.get("idx", i)
        translated = translations.get(idx)
        if translated is None:
            log.warning(f"translate: seg {idx} missing from Gemini response, using empty string")
            translated = ""
        out.append({**seg, "translated_text": translated})

    log.info("translate: done")
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
_RETRY_DELAY = 5  # seconds, doubled on each attempt


def _translate_batch(segments: list[dict], target_lang: str) -> dict[int, str]:
    """Send all segments to Gemini in one call. Returns {idx: translated_text}.
    Retries up to _MAX_RETRIES times on transient errors with exponential backoff."""
    prompt = f"""You are an expert podcast translator. Translate each segment into {target_lang}.

- Translate ONLY the "text" field of each segment.
- Preserve tone, speaker style, and idiomatic expressions.

Segments:
{json.dumps(segments, ensure_ascii=False)}"""

    last_exc = None
    delay = _RETRY_DELAY
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = _client.models.generate_content(
                model=config.GEMINI_MODEL,
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
            return out
        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES:
                log.warning(f"translate: Gemini call failed (attempt {attempt}/{_MAX_RETRIES}): {exc} — retrying in {delay}s")
                time.sleep(delay)
                delay *= 2
            else:
                log.error(f"translate: Gemini call failed after {_MAX_RETRIES} attempts: {exc}")

    raise last_exc
