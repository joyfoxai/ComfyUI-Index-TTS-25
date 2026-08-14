#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMFYUI_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
VERSION="${1:-all}"

if [[ -n "${INDEXTTS_MODELS_DIR:-}" ]]; then
    if [[ "$(basename -- "${INDEXTTS_MODELS_DIR}")" == "indextts" ]]; then
        MODEL_ROOT="${INDEXTTS_MODELS_DIR}"
    else
        MODEL_ROOT="${INDEXTTS_MODELS_DIR}/indextts"
    fi
elif [[ -d /root/comfyui_models ]]; then
    MODEL_ROOT="/root/comfyui_models/indextts"
else
    MODEL_ROOT="${COMFYUI_DIR}/models/indextts"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

case "${VERSION}" in
    2|2.0|2.5|all) ;;
    *)
        echo "Usage: $0 [2|2.5|all]"
        echo "Optional: INDEXTTS_MODELS_DIR=/path/to/models PYTHON_BIN=/path/to/python"
        exit 2
        ;;
esac

mkdir -p "${MODEL_ROOT}"
echo "IndexTTS model root: ${MODEL_ROOT}"

"${PYTHON_BIN}" - "${VERSION}" "${MODEL_ROOT}" <<'PY'
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit(
        "huggingface_hub is missing. Install it with: "
        f"{sys.executable} -m pip install -U huggingface_hub"
    ) from exc

version, model_root = sys.argv[1], Path(sys.argv[2]).expanduser().resolve()
models = {
    "2": ("IndexTeam/IndexTTS-2", "IndexTTS-2"),
    "2.5": ("IndexTeam/IndexTTS-2.5", "IndexTTS-2.5"),
}
selected = list(models) if version == "all" else ["2" if version == "2.0" else version]

for item in selected:
    repo_id, directory_name = models[item]
    target = model_root / directory_name
    target.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {repo_id} -> {target}", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=str(target), resume_download=True)
    print(f"Finished: {repo_id}", flush=True)

required = {
    "2": ("config.yaml", "gpt.pth", "s2mel.pth", "bpe.model", "wav2vec2bert_stats.pt"),
    "2.5": ("config.yaml", "gpt.pth", "s2mel.pth", "codec.pth", "wav2vec2bert_stats.pt"),
}
for item in selected:
    target = model_root / models[item][1]
    missing = [name for name in required[item] if not (target / name).is_file()]
    if missing:
        raise SystemExit(f"Incomplete model at {target}; missing: {', '.join(missing)}")
    print(f"Verified IndexTTS {item}: {target}")
PY

echo
echo "Download complete. Restart ComfyUI, then select the model directory in the Loader node."
