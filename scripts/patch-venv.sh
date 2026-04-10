#!/usr/bin/env bash
# Patch installed TTS package to work with transformers 5.x.
# Run once after creating/rebuilding the venv:
#   ./scripts/patch-venv.sh
set -euo pipefail

SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
echo "==> Site packages: $SITE"

# ── 1. stream_generator.py: guard removed beam-search imports ─────────────────
STREAM="$SITE/TTS/tts/layers/xtts/stream_generator.py"
if grep -q "BeamSearchScorer," "$STREAM" 2>/dev/null; then
  python3 - <<PYEOF
path = "$STREAM"
with open(path) as f:
    content = f.read()

old = """from transformers import (
    BeamSearchScorer,
    ConstrainedBeamSearchScorer,
    DisjunctiveConstraint,
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    PhrasalConstraint,
    PreTrainedModel,
    StoppingCriteriaList,
)"""

new = """from transformers import (
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    PreTrainedModel,
    StoppingCriteriaList,
)
try:
    from transformers import (
        BeamSearchScorer,
        ConstrainedBeamSearchScorer,
        DisjunctiveConstraint,
        PhrasalConstraint,
    )
except ImportError:
    BeamSearchScorer = None
    ConstrainedBeamSearchScorer = None
    DisjunctiveConstraint = None
    PhrasalConstraint = None"""

if old in content:
    with open(path, "w") as f:
        f.write(content.replace(old, new))
    print("  stream_generator: BeamSearchScorer patch applied")
else:
    print("  stream_generator: BeamSearchScorer already patched")
PYEOF
else
  echo "  stream_generator: BeamSearchScorer already patched"
fi

# ── 2. stream_generator.py: guard removed SampleOutput import ────────────────
if grep -q "SampleOutput" "$STREAM" 2>/dev/null; then
  python3 - <<PYEOF
path = "$STREAM"
with open(path) as f:
    content = f.read()

old = "from transformers.generation.utils import GenerateOutput, SampleOutput, logger"
new = """from transformers.generation.utils import GenerateOutput, logger
try:
    from transformers.generation.utils import SampleOutput
except ImportError:
    SampleOutput = GenerateOutput"""

if old in content:
    with open(path, "w") as f:
        f.write(content.replace(old, new))
    print("  stream_generator: SampleOutput patch applied")
else:
    print("  stream_generator: SampleOutput already patched")
PYEOF
fi

# ── 3. gpt_inference.py: add GenerationMixin to GPT2InferenceModel ───────────
GPT_INF="$SITE/TTS/tts/layers/xtts/gpt_inference.py"
if ! grep -q "GenerationMixin" "$GPT_INF" 2>/dev/null; then
  python3 - <<PYEOF
path = "$GPT_INF"
with open(path) as f:
    content = f.read()

old_import = "from transformers import GPT2PreTrainedModel"
new_import = "from transformers import GPT2PreTrainedModel\nfrom transformers import GenerationMixin"
old_class = "class GPT2InferenceModel(GPT2PreTrainedModel):"
new_class = "class GPT2InferenceModel(GPT2PreTrainedModel, GenerationMixin):"

if old_import in content and old_class in content:
    content = content.replace(old_import, new_import)
    content = content.replace(old_class, new_class)
    with open(path, "w") as f:
        f.write(content)
    print("  gpt_inference: GenerationMixin patch applied")
else:
    print("  gpt_inference: already patched or pattern not found")
PYEOF
else
  echo "  gpt_inference: GenerationMixin already patched"
fi

echo "==> Done"
