#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# RelIntel — Hugging Face Spaces Deploy Script
#
# Usage (from repo root):
#   bash deploy_spaces.sh YOUR_HF_USERNAME          # Windows / Git Bash
#   ./deploy_spaces.sh YOUR_HF_USERNAME             # Linux / macOS
#
#   Optional: PYTHON=/path/to/python bash deploy_spaces.sh ...
#
# Prerequisites:
#   pip install huggingface_hub   (in .venv)
#   hf auth login                 (one-time — saves token to ~/.cache/huggingface)
#
# What this does:
#   1. Creates the HF Space (if it doesn't exist)
#   2. Copies the Space README (with YAML front matter) to README.md
#   3. Renames requirements_spaces.txt → requirements.txt for the Space
#   4. Pushes: app.py, src/, data/, requirements.txt, README.md
#   5. Prints the live URL
#
# The data/ directory is committed to the Space repo so ChromaDB and the
# embedder are available at startup — no rebuild needed.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer project venv over pyenv/WSL shims (avoids "required file not found")
resolve_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    if [[ -x "$PYTHON" ]] || command -v "$PYTHON" &>/dev/null; then
      echo "$PYTHON"
      return
    fi
    echo "ERROR: PYTHON is set but not executable: $PYTHON" >&2
    exit 1
  fi
  if [[ -x "$SCRIPT_DIR/.venv/Scripts/python.exe" ]]; then
    echo "$SCRIPT_DIR/.venv/Scripts/python.exe"
    return
  fi
  if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    echo "$SCRIPT_DIR/.venv/bin/python"
    return
  fi
  for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
      echo "$cmd"
      return
    fi
  done
  echo "ERROR: No Python found. Create .venv or set PYTHON=/path/to/python" >&2
  exit 1
}

PYTHON="$(resolve_python)"
echo "Using Python: $PYTHON"

# Windows python.exe cannot read WSL /tmp or /mnt/c paths reliably — convert when needed
to_native_path() {
  local p="$1"
  if [[ "$PYTHON" == *python.exe ]]; then
    if command -v wslpath &>/dev/null; then
      wslpath -w "$p"
    elif command -v cygpath &>/dev/null; then
      cygpath -w "$p"
    else
      (cd "$p" 2>/dev/null && pwd -W) || echo "$p"
    fi
  else
    echo "$p"
  fi
}

HF_USERNAME="${1:-}"
SPACE_NAME="relintel"
SPACE_ID="${HF_USERNAME}/${SPACE_NAME}"

if [[ -z "$HF_USERNAME" ]]; then
  echo "Usage: ./deploy_spaces.sh YOUR_HF_USERNAME"
  exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RelIntel → HF Spaces"
echo "  Space: https://huggingface.co/spaces/${SPACE_ID}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. Create Space (idempotent) ──────────────────────────────────────────────
"$PYTHON" - <<PYEOF
from huggingface_hub import HfApi
api = HfApi()
try:
    api.create_repo(
        repo_id   = "${SPACE_ID}",
        repo_type = "space",
        space_sdk = "gradio",
        private   = False,
        exist_ok  = True,
    )
    print("Space ready: https://huggingface.co/spaces/${SPACE_ID}")
except Exception as e:
    print(f"Space create: {e}")
PYEOF

# ── 2. Stage files in repo (not /tmp — Windows python.exe must see this path) ─
STAGE_DIR="$SCRIPT_DIR/.deploy_stage"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"
trap 'rm -rf "$STAGE_DIR"' EXIT

cp app.py                 "$STAGE_DIR/app.py"
cp README_spaces.md       "$STAGE_DIR/README.md"         # HF reads README.md for card
cp requirements_spaces.txt "$STAGE_DIR/requirements.txt"  # HF installs requirements.txt

# Source files (retriever + generator only — no eval/ingest needed at runtime)
mkdir -p "$STAGE_DIR/src"
cp src/retriever.py  "$STAGE_DIR/src/"
cp src/generator.py  "$STAGE_DIR/src/"

# Data artifacts (pre-built — no rebuild needed on Space)
mkdir -p "$STAGE_DIR/data"
cp data/chunks.json    "$STAGE_DIR/data/"
cp data/embedder.pkl   "$STAGE_DIR/data/"
cp data/companies.json "$STAGE_DIR/data/"
cp data/contacts.json  "$STAGE_DIR/data/"
cp data/deals.json     "$STAGE_DIR/data/"
cp data/interactions.json "$STAGE_DIR/data/"
cp -r data/chroma      "$STAGE_DIR/data/"

STAGED_COUNT=$(find "$STAGE_DIR" -type f | wc -l | tr -d ' ')
echo "Staged ${STAGED_COUNT} files in .deploy_stage/"
if [[ "$STAGED_COUNT" -eq 0 ]]; then
  echo "ERROR: Nothing staged — check that app.py and data/ exist." >&2
  exit 1
fi

STAGE_NATIVE="$(to_native_path "$STAGE_DIR")"
echo "Upload path: $STAGE_NATIVE"

# ── 3. Push ───────────────────────────────────────────────────────────────────
echo ""
echo "Pushing to ${SPACE_ID}..."
"$PYTHON" - <<PYEOF
from huggingface_hub import HfApi
import pathlib
import sys

api   = HfApi()
stage = pathlib.Path(r"${STAGE_NATIVE}")
space = "${SPACE_ID}"

if not stage.is_dir():
    print(f"ERROR: Staging dir not visible to Python: {stage}", file=sys.stderr)
    sys.exit(1)

files = [f for f in stage.rglob("*") if f.is_file()]
if not files:
    print(f"ERROR: No files under {stage}", file=sys.stderr)
    sys.exit(1)

for f in sorted(files):
    path_in_repo = f.relative_to(stage).as_posix()
    api.upload_file(
        path_or_fileobj = str(f),
        path_in_repo    = path_in_repo,
        repo_id         = space,
        repo_type       = "space",
    )
    print(f"  + {path_in_repo}")

print(f"\nPushed {len(files)} files.")
PYEOF

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Deploy complete"
echo ""
echo "  Space URL:  https://huggingface.co/spaces/${SPACE_ID}"
echo ""
echo "  Next step — add your API key:"
echo "  1. Go to https://huggingface.co/spaces/${SPACE_ID}/settings"
echo "  2. Scroll to 'Repository secrets'"
echo "  3. Add:  ANTHROPIC_API_KEY = sk-ant-..."
echo "  4. The Space will restart automatically (~60s)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
