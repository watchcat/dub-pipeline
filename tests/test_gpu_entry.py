# tests/test_gpu_entry.py
from unittest.mock import patch
import pytest
from src.workers import gpu_entry

def test_entry_runs_prep(monkeypatch):
    monkeypatch.setenv("STEP", "prep")
    with patch("src.workers.gpu_entry.gpu_prep.main") as prep, \
         patch("src.workers.gpu_entry.gpu_synth.main") as synth:
        gpu_entry.main()
    prep.assert_called_once(); synth.assert_not_called()

def test_entry_runs_synth(monkeypatch):
    monkeypatch.setenv("STEP", "synth")
    with patch("src.workers.gpu_entry.gpu_prep.main") as prep, \
         patch("src.workers.gpu_entry.gpu_synth.main") as synth:
        gpu_entry.main()
    synth.assert_called_once(); prep.assert_not_called()

def test_entry_unknown_step_exits(monkeypatch):
    monkeypatch.setenv("STEP", "bogus")
    with pytest.raises(SystemExit) as e:
        gpu_entry.main()
    assert e.value.code == 2
