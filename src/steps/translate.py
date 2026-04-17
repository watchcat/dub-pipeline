"""Step 4 — Translate segments with Gemini Flash via the Google GenAI SDK.

Segments are sent in batches with sliding context: the previous 2 already-
translated segments and the next 1 source segment. The system prompt instructs
Gemini to produce natural spoken {target_language} suitable for TTS voice
cloning, not a literal document translation.
"""
import json
import logging
import time

from google import genai
from google.genai import types

from src import config

log = logging.getLogger(__name__)

_client = genai.Client(api_key=config.GEMINI_API_KEY)

_LANG_NAMES = {
    "en": "English",  "es": "Spanish",  "fr": "French",   "de": "German",
    "it": "Italian",  "pt": "Portuguese","pl": "Polish",   "tr": "Turkish",
    "ru": "Russian",  "nl": "Dutch",    "cs": "Czech",    "zh": "Chinese",
    "ja": "Japanese", "hu": "Hungarian", "ko": "Korean",
}

# Static part of the prompt — {source_language}, {target_language}, {batch_size}
# are replaced with str.replace() at dispatch time (avoids escaping JSON braces).
_SYSTEM_PROMPT_TEMPLATE = """\
# Task: Adapt translated podcast transcripts for natural spoken delivery

You are translating a podcast transcript from {source_language} to {target_language}. \
The translated text will be fed directly into a text-to-speech system that clones the \
original speaker's voice. Your translation must sound like natural spoken {target_language}, \
not written {target_language}.

## Critical distinction

This is NOT a document translation. It is a spoken performance translation. A literal, \
grammatically-perfect translation will sound robotic, formal, and obviously dubbed when spoken \
aloud. Your job is to produce text that, when read by a voice actor, would be indistinguishable \
from a native {target_language} podcaster speaking naturally.

## Rules

### 1. Match the register of spoken {target_language}, not written
- Use contractions and elisions that native speakers actually use in speech
- Prefer shorter, simpler sentences over long subordinate clauses, even if the original has them
- Use the conversational vocabulary a podcaster would use, not the formal vocabulary a journalist would write
- Avoid bookish connectors (e.g. in Russian avoid "таким образом", "следовательно", "в связи с этим" — use "значит", "короче", "в общем", "то есть")
- Avoid literal translations of English discourse markers that sound foreign when rendered word-for-word

### 2. Localize idioms, references, and cultural context
- English idioms translated literally sound absurd. Replace with the equivalent idiom in {target_language}, or paraphrase the meaning in natural speech
- Cultural references the target audience won't recognize (specific US brands, regional foods, local politicians, sports references) should be either (a) replaced with a target-culture equivalent if one exists, (b) briefly contextualized in-line, or (c) generalized — pick whichever preserves the speaker's point with least disruption
- Units: convert miles/pounds/Fahrenheit to metric if target language conventions expect metric
- Names of well-known entities: use the form natively used in {target_language}

### 3. Preserve the speaker's voice and personality
- If the original speaker is casual, swears, is sarcastic, rambles, or uses specific verbal tics — preserve that character. Do not sanitize
- If the original is formal and measured, don't make it casual
- Match their energy level and pacing style in word choice

### 4. Preserve timing-adjacent features
- Keep roughly the same number of syllables per segment where possible, but NEVER sacrifice naturalness for length matching. A natural-sounding longer translation beats a stilted shorter one
- If the original has a pause-and-emphasis structure ("It was... terrible."), preserve it with target-language punctuation that cues the same delivery
- Preserve emphasis: if the original stressed a word, ensure the translation places the equivalent emphasis on the translated equivalent (in target-language word order)

### 5. Handle filler words and false starts
- Minor fillers that carry no meaning ("um", "uh", "like", "you know") can be dropped if they don't affect the speaker's character
- BUT if a filler carries meaning or personality ("I mean...", "honestly...", "look,") — translate it with the equivalent natural {target_language} filler
- False starts and self-corrections ("I went to the — actually, we both went to the store") should usually be preserved because they're part of the spoken rhythm, unless they make the translation incomprehensible

### 6. Formatting for TTS
- Output plain text only — no markdown, no quotes around the translation, no segment labels
- Numbers: write out numbers the way a native speaker would say them, spelled out in {target_language} conventions
- Acronyms: if pronounced as letters in {target_language}, keep as letters; if pronounced as a word, write as the word
- Avoid characters the TTS might mispronounce: no ellipsis "…" (use three periods or a comma), no em-dashes for parentheticals (use commas or parentheses in {target_language} conventions), no smart quotes

### 7. Same-language edge case
- If {source_language} == {target_language}: return the original text verbatim, unchanged, no paraphrasing

## Context handling

You will receive segments in batches of up to {batch_size}. For each batch you also see:
- The previous 2 segments (already translated) — use these to maintain terminology, tone, and flow consistency
- The next 1 segment (in source language) — use this to understand where the current sentence is heading, especially if a sentence spans segments
- The speaker ID — if the speaker changes between segments, the register may shift (e.g. host to guest)

Translate only the segments marked TRANSLATE. Do not re-translate context segments.

## Input format

PREVIOUS (already translated): [text]
PREVIOUS (already translated): [text]
TRANSLATE [idx=42] [SPEAKER_00]: [source text]
TRANSLATE [idx=43] [SPEAKER_00]: [source text]
TRANSLATE [idx=44] [SPEAKER_01]: [source text]
NEXT (source only): [text]

## Output format

Return a JSON array, one object per TRANSLATE segment, in the same order:

[
  {"idx": 42, "translated_text": "..."},
  {"idx": 43, "translated_text": "..."},
  {"idx": 44, "translated_text": "..."}
]

No prose before or after the JSON. No code fences. Just the array.

## Quality check before responding

Before returning, read your translation aloud in your head. Ask:
1. Would a native {target_language} podcaster actually say this, or does it sound like a translation?
2. Does it preserve the speaker's energy and personality?
3. Are there any words that would trip up a TTS system?
If yes to any sounding wrong — rewrite it.\
"""

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
_RETRY_DELAY = 10   # seconds, doubled on each retry
_FALLBACK_MODEL = "gemini-2.0-flash"
_BATCH_SIZE = 50    # segments per Gemini call


def translate(segments: list[dict], source_lang: str, target_lang: str) -> list[dict]:
    """
    Translate all segments from source_lang to target_lang using Gemini Flash.
    Returns segments with "translated_text" added.
    Same-language jobs copy text verbatim.
    """
    if source_lang.lower() == target_lang.lower():
        log.info(f"translate: source == target ({source_lang}), copying text verbatim")
        return [{**seg, "translated_text": seg["text"]} for seg in segments]

    src_name = _LANG_NAMES.get(source_lang.lower(), source_lang)
    tgt_name = _LANG_NAMES.get(target_lang.lower(), target_lang)
    log.info(f"translate: {len(segments)} segments, {src_name} → {tgt_name} via Gemini")

    system_prompt = (
        _SYSTEM_PROMPT_TEMPLATE
        .replace("{source_language}", src_name)
        .replace("{target_language}", tgt_name)
        .replace("{batch_size}", str(_BATCH_SIZE))
    )

    input_segs = [
        {
            "idx":     seg.get("idx", i),
            "text":    seg.get("text", "").strip(),
            "speaker": seg.get("speaker", ""),
        }
        for i, seg in enumerate(segments)
    ]

    translations: dict[int, str] = {}
    recent_translated: list[dict] = []  # {"text": src, "translated_text": tgt}

    for batch_start in range(0, len(input_segs), _BATCH_SIZE):
        batch = input_segs[batch_start : batch_start + _BATCH_SIZE]
        prev_ctx  = recent_translated[-2:]
        next_idx  = batch_start + len(batch)
        next_seg  = input_segs[next_idx] if next_idx < len(input_segs) else None

        log.info(
            f"translate: batch {batch_start // _BATCH_SIZE + 1}, "
            f"segs {batch_start}–{batch_start + len(batch) - 1}"
        )
        batch_result = _translate_batch(batch, system_prompt, prev_ctx, next_seg)
        translations.update(batch_result)

        for seg in batch:
            recent_translated.append({
                "text":            seg["text"],
                "translated_text": batch_result.get(seg["idx"], ""),
            })

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
        log.info(
            f"translate: sample — '{out[0].get('text','')[:60]}' "
            f"→ '{out[0].get('translated_text','')[:60]}'"
        )
    return out


def _build_user_message(
    batch: list[dict],
    prev_ctx: list[dict],
    next_seg: dict | None,
) -> str:
    lines = []
    for ctx in prev_ctx:
        lines.append(f"PREVIOUS (already translated): {ctx['translated_text']}")
    for seg in batch:
        speaker_tag = f" [{seg['speaker']}]" if seg.get("speaker") else ""
        lines.append(f"TRANSLATE [idx={seg['idx']}]{speaker_tag}: {seg['text']}")
    if next_seg:
        lines.append(f"NEXT (source only): {next_seg['text']}")
    return "\n".join(lines)


def _translate_batch(
    segments: list[dict],
    system_prompt: str,
    prev_ctx: list[dict],
    next_seg: dict | None,
) -> dict[int, str]:
    """Send one batch to Gemini. Returns {idx: translated_text}.
    Retries up to _MAX_RETRIES times with exponential backoff, then falls back
    to _FALLBACK_MODEL."""
    user_message = _build_user_message(segments, prev_ctx, next_seg)

    models = [config.GEMINI_MODEL, _FALLBACK_MODEL]
    last_exc = None
    for model in models:
        delay = _RETRY_DELAY
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = _client.models.generate_content(
                    model=model,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=_RESPONSE_SCHEMA,
                    ),
                )
                result = json.loads(response.text)
                out: dict[int, str] = {}
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
                    log.warning(
                        f"translate: {model} attempt {attempt}/{_MAX_RETRIES}: "
                        f"{exc} — retrying in {delay}s"
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    log.error(f"translate: {model} failed all retries: {exc} — trying fallback")

    raise last_exc
