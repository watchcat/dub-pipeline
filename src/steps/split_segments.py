"""Split segments that exceed MAX_SEGMENT_SEC at sentence boundaries.

WhisperX produces word-level timestamps so we can find natural break points:
  1. Words ending with sentence-final punctuation (. ? ! …)
  2. Pauses between consecutive words (gap > PAUSE_THRESHOLD)
  3. Hard cut at MAX_SEGMENT_SEC if no natural break was found

Splitting runs BEFORE translation so each sub-segment gets its own
DeepL call. Indices are renumbered sequentially across the full segment
list after splitting.

Returns a new list of segments with integer `idx` values.
"""
import logging

log = logging.getLogger(__name__)

MAX_SEGMENT_SEC  = 30.0   # segments longer than this are split
MIN_CHUNK_SEC    = 5.0    # don't create chunks shorter than this
PAUSE_THRESHOLD  = 0.4    # seconds of silence = acceptable break point
SENTENCE_ENDINGS = {".", "?", "!", "…", "。", "！", "？"}


def split_long_segments(segments: list[dict]) -> list[dict]:
    """
    For each segment longer than MAX_SEGMENT_SEC, split at sentence/pause
    boundaries using word-level timestamps.
    Returns a new flat list with sequential integer indices.
    """
    out = []
    for seg in segments:
        dur = seg["end_sec"] - seg["start_sec"]
        if dur <= MAX_SEGMENT_SEC:
            out.append(seg)
        else:
            chunks = _split(seg)
            log.info(
                f"split_segments: seg {seg['idx']} ({dur:.1f}s) → "
                f"{len(chunks)} chunks"
            )
            out.extend(chunks)

    # Renumber indices sequentially so file naming stays clean
    for i, seg in enumerate(out):
        seg["idx"] = i

    return out


def _split(seg: dict) -> list[dict]:
    words = seg.get("words") or []
    if not words:
        # No word timestamps — can't split; keep as-is (will be skipped in synth)
        log.warning(f"split_segments: seg {seg['idx']} has no word timestamps, cannot split")
        return [seg]

    chunks: list[dict] = []
    current: list[dict] = []
    chunk_start: float = seg["start_sec"]

    def flush(end_time: float):
        nonlocal chunk_start
        if not current:
            return
        text = _words_to_text(current)
        chunks.append({
            **seg,
            "start_sec": chunk_start,
            "end_sec":   end_time,
            "text":      text,
            "words":     list(current),
        })
        current.clear()
        chunk_start = end_time

    for i, word in enumerate(words):
        word_end   = float(word.get("end") or word.get("start") or 0)
        word_start = float(word.get("start") or 0)
        next_word  = words[i + 1] if i + 1 < len(words) else None

        current.append(word)
        current_dur = word_end - chunk_start

        # Determine whether this is a good cut point
        is_sentence_end = _is_sentence_end(word.get("word", ""))
        pause = (float(next_word["start"]) - word_end) if next_word else 0.0
        is_pause = pause >= PAUSE_THRESHOLD

        if current_dur >= MAX_SEGMENT_SEC:
            # Hard cut — we've hit the limit regardless of content
            flush(word_end)
        elif current_dur >= MIN_CHUNK_SEC and (is_sentence_end or is_pause):
            # Natural break and we've accumulated enough
            flush(word_end)

    # Flush remaining words
    if current:
        last_end = float(current[-1].get("end") or current[-1].get("start") or seg["end_sec"])
        flush(last_end)

    return chunks if chunks else [seg]


def _is_sentence_end(word: str) -> bool:
    stripped = word.strip()
    if not stripped:
        return False
    return stripped[-1] in SENTENCE_ENDINGS


def _words_to_text(words: list[dict]) -> str:
    return " ".join(w.get("word", "") for w in words).strip()
