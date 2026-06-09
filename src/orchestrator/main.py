"""Production entrypoint: wire collaborators, start the reconciler, serve uvicorn."""
import threading
import uvicorn
from src import config
from src.orchestrator import app as appmod, nebius
from src.orchestrator.dispatch import HttpDispatcher, NebiusDispatcher
from src.orchestrator.reconciler import Reconciler
from src.orchestrator.reporting import Reporter
from src.orchestrator.runs import PgRunStore


def build():
    store = PgRunStore(config.DATABASE_URL)
    dispatchers = {
        "gpu": NebiusDispatcher(store, nebius_client=nebius),
        "cpu": HttpDispatcher({"text": config.CPU_TEXT_URL, "mux": config.CPU_MUX_URL}),
    }
    sm = appmod.configure(store, dispatchers, Reporter())
    reconciler = Reconciler(store, sm, nebius)
    return store, reconciler


def main():
    store, reconciler = build()
    store.init_schema()
    threading.Thread(target=reconciler.run_forever, daemon=True).start()
    uvicorn.run(appmod.app, host="0.0.0.0", port=8080)


if __name__ == "__main__":
    main()
