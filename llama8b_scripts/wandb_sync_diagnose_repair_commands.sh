#!/bin/bash
set -euo pipefail

# wandb_sync_diagnose_repair_commands.sh
# Purpose: Diagnose and optionally repair Weights & Biases offline run artifacts that can
#          cause `wandb sync` to fail with AssertionError (missing record_type),
#          then optionally attempt syncing.
#
# Usage examples:
#   ./llama8b_scripts/wandb_sync_diagnose_repair_commands.sh --dry-run
#   ./llama8b_scripts/wandb_sync_diagnose_repair_commands.sh --repair --quarantine-bad
#   ./llama8b_scripts/wandb_sync_diagnose_repair_commands.sh --attempt-sync
#   ./llama8b_scripts/wandb_sync_diagnose_repair_commands.sh --upgrade-wandb --wandb-version 0.17.9
#   ./llama8b_scripts/wandb_sync_diagnose_repair_commands.sh --only-run-id 5sfhoq9p --repair --attempt-sync
#
# Notes:
# - Default W&B base URL is set to on-prem server if not provided.
# - Repairs are conservative: only drop the last line if a JSONL file's last line is invalid.
# - Quarantine moves clearly corrupted runs to a separate directory to avoid blocking sync-all.

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

WANDB_ROOT_DEFAULT="/data/home/zdhs0054/zzhou/MaskLLM/wandb"
WANDB_ROOT="${WANDB_ROOT:-$WANDB_ROOT_DEFAULT}"
WANDB_BASE_URL="${WANDB_BASE_URL:-http://wandb.pdzhou.com}"

DRY_RUN=false
DO_REPAIR=false
QUARANTINE_BAD=false
ATTEMPT_SYNC=false
UPGRADE_WANDB=false
WANDB_VERSION=""
ONLY_RUN_ID=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -r, --root-dir DIR         Set W&B runs root directory (default: $WANDB_ROOT)
      --dry-run              Show what would be done without modifying files
      --repair               Apply minimal repairs (trim last line if JSONL is truncated)
      --quarantine-bad       Move clearly corrupted runs to _quarantine/
      --attempt-sync         Run 'wandb sync --sync-all' after diagnostics/repairs
      --only-run-id ID       Restrict to a single run id (e.g., 5sfhoq9p)
      --upgrade-wandb        pip install --upgrade wandb (uses Tsinghua mirror)
      --wandb-version VER    Pin wandb version when upgrading (e.g., 0.17.9)
  -h, --help                 Show this help

Environment:
  WANDB_ROOT         Root dir containing W&B run folders (offline-run-*, run-*)
  WANDB_BASE_URL     Base URL for W&B server (default: $WANDB_BASE_URL)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--root-dir)
      WANDB_ROOT="$2"; shift 2;;
    --dry-run)
      DRY_RUN=true; shift;;
    --repair)
      DO_REPAIR=true; shift;;
    --quarantine-bad)
      QUARANTINE_BAD=true; shift;;
    --attempt-sync)
      ATTEMPT_SYNC=true; shift;;
    --only-run-id)
      ONLY_RUN_ID="$2"; shift 2;;
    --upgrade-wandb)
      UPGRADE_WANDB=true; shift;;
    --wandb-version)
      WANDB_VERSION="$2"; shift 2;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1" >&2; usage; exit 2;;
  esac
done

log() { echo "[$(date +"%F %T")] $*"; }
run_cmd() {
  if $DRY_RUN; then
    echo "DRY-RUN: $*"
  else
    eval "$@"
  fi
}

ensure_dirs() {
  mkdir -p "$WANDB_ROOT"
  mkdir -p "$WANDB_ROOT/_quarantine"
}

python_json_check() {
  # Args: file_path
  python3 - "$1" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path, 'rb') as f:
        last = None
        for line in f:
            if line.strip():
                last = line
        if last is None:
            print("EMPTY", end='')
            sys.exit(0)
        try:
            json.loads(last.decode('utf-8', errors='strict'))
            print("OK", end='')
        except Exception:
            print("BAD", end='')
except FileNotFoundError:
    print("MISSING", end='')
PY
}

trim_last_line() {
  # Args: file_path
  local file="$1"
  if $DRY_RUN; then
    echo "DRY-RUN: sed -i '$d' '$file'"
  else
    # Use a safe approach to drop exactly the last line
    tmp_file="${file}.tmp.${TIMESTAMP}"
    head -n -1 "$file" > "$tmp_file" || true
    mv "$tmp_file" "$file"
  fi
}

diagnose_run_dir() {
  # Args: run_dir
  local run_dir="$1"
  local run_id
  run_id=$(basename "$run_dir")
  local issues=()
  local actions=()

  # Identify typical wandb files
  local wandb_bin="$run_dir/media/graph.qtf" # placeholder name; .wandb may exist at top-level too
  local wandb_file_candidates=(
    "$run_dir/run-*.wandb"
    "$run_dir/*.wandb"
  )
  local hist_file="$run_dir/wandb-history.jsonl"
  local events_file="$run_dir/wandb-events.jsonl"

  # Check .wandb files (if any) for zero size
  local wandb_files=( )
  for pat in "${wandb_file_candidates[@]}"; do
    for f in $pat; do
      if [[ -f "$f" ]]; then wandb_files+=("$f"); fi
    done
  done
  if [[ ${#wandb_files[@]} -gt 0 ]]; then
    for f in "${wandb_files[@]}"; do
      if [[ ! -s "$f" ]]; then
        issues+=("zero_byte_wandb:$f")
      fi
    done
  fi

  # Check history/events JSONL last line validity
  if [[ -f "$hist_file" ]]; then
    local res
    res=$(python_json_check "$hist_file") || true
    if [[ "$res" == "BAD" ]]; then
      issues+=("history_last_line_bad:$hist_file")
      actions+=("trim_last_line:$hist_file")
    elif [[ "$res" == "EMPTY" ]]; then
      issues+=("history_empty:$hist_file")
    fi
  fi
  if [[ -f "$events_file" ]]; then
    local res
    res=$(python_json_check "$events_file") || true
    if [[ "$res" == "BAD" ]]; then
      issues+=("events_last_line_bad:$events_file")
      actions+=("trim_last_line:$events_file")
    elif [[ "$res" == "EMPTY" ]]; then
      issues+=("events_empty:$events_file")
    fi
  fi

  # Heuristic: if there are zero-byte .wandb files, mark run as bad
  local mark_bad=false
  for it in "${issues[@]}"; do
    if [[ "$it" == zero_byte_wandb:* ]]; then mark_bad=true; fi
  done

  echo "RUN $run_id"
  if [[ ${#issues[@]} -eq 0 ]]; then
    echo "  - No obvious issues found"
  else
    for it in "${issues[@]}"; do echo "  - Issue: $it"; done
  fi

  # Repairs
  if $DO_REPAIR; then
    for act in "${actions[@]}"; do
      if [[ "$act" == trim_last_line:* ]]; then
        local file=${act#trim_last_line:}
        log "Trimming last line (invalid JSON) in $file"
        trim_last_line "$file"
      fi
    done
  fi

  # Quarantine when clearly bad
  if $QUARANTINE_BAD && $mark_bad; then
    local dest="$WANDB_ROOT/_quarantine/${run_id}_${TIMESTAMP}"
    log "Quarantining bad run $run_id -> $dest"
    run_cmd "mv '$run_dir' '$dest'"
  fi
}

attempt_sync_all() {
  log "Attempting wandb sync --sync-all"
  local cmd="WANDB_BASE_URL='$WANDB_BASE_URL' wandb sync --sync-all"
  run_cmd "$cmd"
}

attempt_sync_one() {
  # Args: run_dir
  local run_dir="$1"
  log "Attempting wandb sync for $run_dir"
  local cmd="WANDB_BASE_URL='$WANDB_BASE_URL' wandb sync '$run_dir'"
  run_cmd "$cmd"
}

maybe_upgrade_wandb() {
  if ! $UPGRADE_WANDB; then return; fi
  local ver_arg=""
  if [[ -n "$WANDB_VERSION" ]]; then
    ver_arg=="==${WANDB_VERSION}"
  fi
  log "Upgrading wandb${WANDB_VERSION:+ to version $WANDB_VERSION} using Tsinghua mirror"
  # Use Tsinghua mirror for speed/stability
  run_cmd "pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade 'wandb${ver_arg}'"
}

main() {
  ensure_dirs
  log "WANDB_ROOT: $WANDB_ROOT"
  log "WANDB_BASE_URL: $WANDB_BASE_URL"
  log "Python: $(python3 --version 2>/dev/null || echo not found)"
  log "wandb: $(python3 -c 'import wandb,sys;print(wandb.__version__)' 2>/dev/null || echo not found)"

  maybe_upgrade_wandb

  shopt -s nullglob
  # Collect candidate run directories
  local run_dirs=( )
  for d in "$WANDB_ROOT"/offline-run-* "$WANDB_ROOT"/run-*; do
    if [[ -d "$d" ]]; then
      if [[ -n "$ONLY_RUN_ID" ]]; then
        [[ "$(basename "$d")" == *"$ONLY_RUN_ID"* ]] && run_dirs+=("$d") || true
      else
        run_dirs+=("$d")
      fi
    fi
  done

  if [[ ${#run_dirs[@]} -eq 0 ]]; then
    log "No run directories found under $WANDB_ROOT"
  else
    log "Found ${#run_dirs[@]} run directories to check"
  fi

  for rd in "${run_dirs[@]}"; do
    diagnose_run_dir "$rd"
  done

  if $ATTEMPT_SYNC; then
    if [[ -n "$ONLY_RUN_ID" ]]; then
      for rd in "${run_dirs[@]}"; do
        attempt_sync_one "$rd"
      done
    else
      attempt_sync_all
    fi
  fi

  log "Done."
}

main "$@"


