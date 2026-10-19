#!/usr/bin/env bash
# One-command reproduction of every number, figure and table in the report.
# Run `bash run_all.sh --help` for usage.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

usage() {
  cat <<'EOF'
Usage: bash run_all.sh [--quick]
  (no flag)  full run: SIFT 1M + GloVe-100 1.18M                        -> ../results/
  --quick    first 100,000 base vectors, 2,000 queries (< 15 min on a laptop) -> ../results_quick/
Environment: VENV_DIR (default ../.venv), ANNBENCH_DATA_DIR (../data), ANNBENCH_RESULTS_DIR,
             ANNBENCH_BUILD_THREADS (4)
EOF
}

QUICK=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; usage >&2; exit 2 ;;
  esac
done
export ANNBENCH_QUICK="$QUICK"
log() { echo "[run_all] $*"; }
log "mode: $([ "$QUICK" = 1 ] && echo quick || echo full)"

# --- 1. python environment ---------------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-../.venv}"
if [ -z "${VIRTUAL_ENV:-}" ]; then
  if [ ! -d "$VENV_DIR" ]; then
    log "creating virtual environment in $VENV_DIR"
    python3 -m venv "$VENV_DIR"
  fi
  set +u; source "$VENV_DIR/bin/activate"; set -u
fi
if ! python -I -c "import annbench, faiss, hnswlib, h5py, numpy, pandas, matplotlib" 2>/dev/null; then
  log "installing requirements"
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt
  python -m pip install --quiet -e .
fi

# --- 2. data -------------------------------------------------------------------------------------------
bash scripts/download_data.sh
python -I -m annbench.data prepare

# --- 3. machine + library versions ---------------------------------------------------------------------
RESULTS_DIR="$(python -I -c 'from annbench import config; print(config.results_dir())')"
mkdir -p "$RESULTS_DIR"
python -I -m annbench.sysinfo --out "$RESULTS_DIR/system_info.json" > /dev/null

# --- 4. experiments + report assets --------------------------------------------------------------------
# One module per stage in annbench/stages/. Each appends its runs to results.csv through
# annbench.results_log and writes its own figures/tables. Stages that do not exist yet are reported
# loudly instead of being skipped silently.
STAGES=(part1_exact part2_benchmark part3_experiment part4_experiment report_assets)
missing=()
for stage in "${STAGES[@]}"; do
  if [ -f "annbench/stages/${stage}.py" ]; then
    log "stage: $stage"
    python -I -m "annbench.stages.${stage}"
  else
    missing+=("$stage")
  fi
done
if [ ${#missing[@]} -gt 0 ]; then
  log "WARNING: stages not implemented yet: ${missing[*]}"
fi
log "done; outputs in $RESULTS_DIR"
