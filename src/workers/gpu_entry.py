# src/workers/gpu_entry.py
"""GPU container entrypoint: run prep or synth based on the STEP env var."""
import os
import sys
from src.workers import gpu_prep, gpu_synth

_ENTRY = {"prep": "gpu_prep", "synth": "gpu_synth"}


def main() -> None:
    step = os.environ.get("STEP", "")
    module_name = _ENTRY.get(step)
    if module_name is None:
        sys.stderr.write(f"gpu_entry: unknown STEP={step!r}\n")
        raise SystemExit(2)
    module = gpu_prep if module_name == "gpu_prep" else gpu_synth
    module.main()


if __name__ == "__main__":
    main()
