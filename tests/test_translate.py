"""Unit tests for HY-MT prompt building and response parsing."""
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

from src.steps.translate import (
    _build_hymt_prompt,
    _HYMT_TARGET_RE,
)


class TestBuildHymtPrompt:
    def test_basic_prompt_structure(self):
        batch = [
            {"idx": 0, "text": "Hello world", "speaker": "SPEAKER_00"},
            {"idx": 1, "text": "How are you", "speaker": "SPEAKER_01"},
        ]
        prompt = _build_hymt_prompt(batch, "en", "ru", prev_ctx=[], next_seg=None)

        assert "English" in prompt
        assert "Russian" in prompt
        assert "[idx=0] [SPEAKER_00]: Hello world" in prompt
        assert "[idx=1] [SPEAKER_01]: How are you" in prompt
        assert "TRANSLATE:" in prompt
        assert "<target>translated text</target>" in prompt

    def test_with_context(self):
        batch = [{"idx": 5, "text": "Third segment", "speaker": ""}]
        prev_ctx = [
            {"translated_text": "First translated"},
            {"translated_text": "Second translated"},
        ]
        next_seg = {"idx": 6, "text": "Fourth segment", "speaker": ""}

        prompt = _build_hymt_prompt(batch, "en", "es", prev_ctx, next_seg)

        assert "CONTEXT (already translated):" in prompt
        assert "[1] First translated" in prompt
        assert "[2] Second translated" in prompt
        assert "NEXT (for context only):" in prompt
        assert "Fourth segment" in prompt

    def test_no_context(self):
        batch = [{"idx": 0, "text": "First segment", "speaker": ""}]
        prompt = _build_hymt_prompt(batch, "en", "fr", prev_ctx=[], next_seg=None)

        assert "CONTEXT" not in prompt
        assert "NEXT" not in prompt

    def test_no_speaker(self):
        batch = [{"idx": 0, "text": "No speaker here", "speaker": ""}]
        prompt = _build_hymt_prompt(batch, "en", "de", prev_ctx=[], next_seg=None)

        assert "[idx=0]: No speaker here" in prompt
        assert "SPEAKER" not in prompt


class TestTargetRegex:
    def test_parses_target_tags(self):
        response = (
            "[idx=0]: <target>Привет мир</target>\n"
            "[idx=1]: <target>Как дела</target>\n"
        )
        matches = {int(m.group(1)): m.group(2).strip() for m in _HYMT_TARGET_RE.finditer(response)}
        assert matches == {0: "Привет мир", 1: "Как дела"}

    def test_handles_extra_whitespace(self):
        response = "[idx=42]:  <target>  Hola mundo  </target>\n"
        matches = {int(m.group(1)): m.group(2).strip() for m in _HYMT_TARGET_RE.finditer(response)}
        assert matches == {42: "Hola mundo"}

    def test_no_match_returns_empty(self):
        response = "Some random text without target tags"
        matches = {int(m.group(1)): m.group(2).strip() for m in _HYMT_TARGET_RE.finditer(response)}
        assert matches == {}
