from unittest.mock import patch
from src.workers import cpu_text

SEGS = [{"idx": 0, "text": "hi", "start_sec": 0, "end_sec": 40}]
SPLIT = [{"idx": 0, "text": "hi", "start_sec": 0, "end_sec": 20},
         {"idx": 1, "text": "there", "start_sec": 20, "end_sec": 40}]
TRANSLATED = [{**SPLIT[0], "translated_text": "hola"},
              {**SPLIT[1], "translated_text": "ahi"}]

def test_text_splits_then_translates_and_writes_artifact():
    with patch("src.workers.cpu_text.artifacts.read_segments", return_value=SEGS), \
         patch("src.workers.cpu_text.split_segments.split_long_segments", return_value=SPLIT) as sp, \
         patch("src.workers.cpu_text.translate.translate", return_value=TRANSLATED) as tr, \
         patch("src.workers.cpu_text.artifacts.write_segments",
               return_value="dub-runs/r1/segments.json") as wr:
        out = cpu_text.run({"run_id": "r1", "episode_id": 456,
                            "segments_key": "k", "source_lang": "en", "language": "es"})
    sp.assert_called_once_with(SEGS)
    tr.assert_called_once_with(SPLIT, "en", "es")
    assert wr.call_args.args == ("r1", TRANSLATED)
    assert out == {"segments_key": "dub-runs/r1/segments.json"}
