"""Unit tests for HY-MT prompt building."""
import re
from unittest.mock import patch, MagicMock

# Patch config before importing translate
import sys
import os
os.environ.setdefault("PROGRESS_URL", "http://test")
os.environ.setdefault("R2_ENDPOINT", "http://test")
os.environ.setdefault("R2_ACCESS_KEY_ID", "test")
os.environ.setdefault("R2_SECRET_ACCESS_KEY", "test")
os.environ.setdefault("R2_BUCKET", "test")
os.environ.setdefault("R2_PUBLIC_URL", "http://test")
os.environ.setdefault("GEMINI_API_KEY", "test")
os.environ.setdefault("HF_TOKEN", "test")

from src.steps.translate import _build_hymt_message


class TestBuildHymtMessage:
    def test_english_prompt_for_non_chinese(self):
        messages = _build_hymt_message("Hello world", "es", "en")
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "Translate the following segment into Spanish" in messages[0]["content"]
        assert "without additional explanation" in messages[0]["content"]
        assert "Hello world" in messages[0]["content"]

    def test_chinese_prompt_when_target_is_zh(self):
        messages = _build_hymt_message("Hello world", "zh", "en")
        assert "翻译为" in messages[0]["content"]
        assert "Hello world" in messages[0]["content"]

    def test_chinese_prompt_when_source_is_zh(self):
        messages = _build_hymt_message("你好世界", "en", "zh")
        assert "翻译为" in messages[0]["content"]
        assert "English" in messages[0]["content"]
        assert "你好世界" in messages[0]["content"]

    def test_non_chinese_pair(self):
        messages = _build_hymt_message("Bonjour le monde", "de", "fr")
        assert "Translate the following segment into German" in messages[0]["content"]
        assert "Bonjour le monde" in messages[0]["content"]

    def test_preserves_source_text(self):
        text = "This is a test with special chars: <>&"
        messages = _build_hymt_message(text, "ru", "en")
        assert text in messages[0]["content"]
