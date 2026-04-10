"""Step 4 — Translate segments with DeepL Pro (batch, with context).

Returns segments with an added "translated_text" field.
If source == target language, copies text without calling DeepL.
"""
import logging
import time
import deepl
from src import config

log = logging.getLogger(__name__)

_translator = None

_DEEPL_MAX_RETRIES = 3
_DEEPL_RETRY_DELAY = 60  # seconds


def _get_translator() -> deepl.Translator:
    global _translator
    if _translator is None:
        _translator = deepl.Translator(config.DEEPL_API_KEY)
    return _translator


def translate(segments: list[dict], source_lang: str, target_lang: str) -> list[dict]:
    """
    Translate all segment texts from source_lang to target_lang.
    Uses DeepL Pro context parameter: each chunk gets the previous segment
    text as context for better continuity.
    If source_lang == target_lang, copies text directly without calling DeepL.
    Returns segments with "translated_text" added.
    """
    # Same-language shortcut — skip DeepL entirely
    if source_lang.lower() == target_lang.lower():
        log.info(f"translate: source == target ({source_lang}), copying text verbatim")
        return [{**seg, "translated_text": seg["text"]} for seg in segments]

    translator = _get_translator()
    src = source_lang.upper()
    tgt = _deepl_target_lang(target_lang)

    log.info(f"translate: {len(segments)} segments, {source_lang} → {target_lang}")

    texts = [s["text"] for s in segments]
    translated = _batch_translate(translator, texts, src, tgt)

    out = [{**seg, "translated_text": tr} for seg, tr in zip(segments, translated)]
    log.info("translate: done")
    return out


def _batch_translate(
    translator: deepl.Translator,
    texts: list[str],
    source_lang: str,
    target_lang: str,
) -> list[str]:
    """Translate texts in chunks of 50 (DeepL array limit), with retry on quota error."""
    CHUNK = 50
    results = []
    for i in range(0, len(texts), CHUNK):
        chunk = texts[i : i + CHUNK]
        context = texts[i - 1] if i > 0 else None
        results.extend(_translate_chunk(translator, chunk, source_lang, target_lang, context))
    return results


def _translate_chunk(
    translator: deepl.Translator,
    chunk: list[str],
    source_lang: str,
    target_lang: str,
    context: str | None,
) -> list[str]:
    kwargs = dict(
        text=chunk,
        source_lang=source_lang,
        target_lang=target_lang,
        preserve_formatting=True,
    )
    if context:
        kwargs["context"] = context

    for attempt in range(1, _DEEPL_MAX_RETRIES + 1):
        try:
            result = translator.translate_text(**kwargs)
            return [r.text for r in result]
        except deepl.QuotaExceededException:
            if attempt < _DEEPL_MAX_RETRIES:
                log.warning(f"translate: DeepL quota exceeded, retrying in {_DEEPL_RETRY_DELAY}s (attempt {attempt}/{_DEEPL_MAX_RETRIES})")
                time.sleep(_DEEPL_RETRY_DELAY)
            else:
                raise
        except deepl.DeepLException as e:
            log.error(f"translate: DeepL error on attempt {attempt}: {e}")
            raise


def _deepl_target_lang(lang: str) -> str:
    """Map ISO 639-1 codes to DeepL target language codes."""
    mapping = {
        "en": "EN-US",
        "pt": "PT-BR",
        "zh": "ZH-HANS",
    }
    return mapping.get(lang.lower(), lang.upper())
