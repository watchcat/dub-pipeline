# tests/test_main.py
from unittest.mock import patch
from src.orchestrator import main as m

def test_build_wires_app_and_reconciler(monkeypatch):
    monkeypatch.setattr(m.config, "DATABASE_URL", "postgres://x")
    monkeypatch.setattr(m.config, "CPU_TEXT_URL", "http://t/run")
    monkeypatch.setattr(m.config, "CPU_MUX_URL", "http://mx/run")
    with patch("src.orchestrator.main.PgRunStore") as Store, \
         patch("src.orchestrator.main.appmod.configure") as cfg:
        store, reconciler = m.build()
    Store.assert_called_once_with("postgres://x")
    _, dispatchers, _ = cfg.call_args.args
    assert set(dispatchers) == {"gpu", "cpu"}
    assert reconciler.store is store
