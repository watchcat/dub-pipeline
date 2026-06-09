from src.orchestrator import auth


def test_token_roundtrip_and_tamper(monkeypatch):
    monkeypatch.setattr(auth.config, "ORCH_CALLBACK_SECRET", "k")
    t = auth.make_token("run1", "prep")
    assert auth.verify_token("run1", "prep", t) is True
    assert auth.verify_token("run1", "synth", t) is False
    assert auth.verify_token("run2", "prep", t) is False
    assert auth.verify_token("run1", "prep", "deadbeef") is False
