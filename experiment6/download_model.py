"""Download a pinned public GPT-2 snapshot without loading or running it."""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "openai-community/gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
FILES = ["config.json", "generation_config.json", "model.safetensors",
         "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]


def main():
    os.environ.setdefault("HF_HOME", str(ROOT / "work/hf-cache"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    from huggingface_hub import snapshot_download
    target = ROOT / "work/models/gpt2"
    snapshot_download(MODEL_ID, revision=REVISION, allow_patterns=FILES,
                      local_dir=target, token=False)
    records = {}
    for name in FILES:
        path = target / name
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        records[name] = {"bytes": path.stat().st_size, "sha256": h.hexdigest()}
    output = ROOT / "outputs/experiment6"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"model_id": MODEL_ID, "revision": REVISION, "files": records,
                "source": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
                "weights_evaluated": False}
    (output / "model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"downloaded_files": len(records), "revision": REVISION,
                      "bytes": sum(x["bytes"] for x in records.values())}))


if __name__ == "__main__":
    main()
