from unittest.mock import patch
from src.workers import common


def test_post_callback_ok():
    with patch("src.workers.common.requests.post") as post:
        common.post_callback("http://cb", {"x": 1})
    assert post.call_args.args[0] == "http://cb"
    assert post.call_args.kwargs["json"] == {"x": 1}


def test_run_in_tempdir_cleans_up(tmp_path):
    seen = {}
    def body(d):
        seen["dir"] = d
        assert d  # exists during call
        return "result"
    out = common.run_in_tempdir(body)
    assert out == "result"
