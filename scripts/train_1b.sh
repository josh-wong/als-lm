#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# ALS-LM 1B Production Training Launch Script
#
# Pre-flight checks, resume detection, tmux guidance, and deepspeed launch.
# Projected training time: ~45 hours for 3 epochs (11,679 steps).
# =============================================================================

# ---- Constants --------------------------------------------------------------

MIN_DISK_GB=50
MIN_WSL_MEM_GB=47
TMUX_SESSION="als-1b-train"
CONFIG="1B"
DATA_DIR="data/tokenized/v1.2.0"
DS_CONFIG="config/ds_zero2.json"
CHECKPOINT_INTERVAL=500
CHECKPOINT_BASE="${ALS_CHECKPOINT_BASE:-checkpoints}"
LOG_DIR="logs"
PREFLIGHT_LOG_DIR="logs/preflight"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${PROJECT_DIR}/.venv"

# ---- Activate virtual environment ------------------------------------------

if [[ -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo "Using active virtual environment: ${VIRTUAL_ENV}"
else
    echo -e "\033[0;31mERROR: No .venv found at ${VENV_DIR} and no active virtual environment.\033[0m"
    echo "Create one with: python3 -m venv .venv && pip install -r requirements.txt"
    exit 1
fi

# ---- Color codes ------------------------------------------------------------

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BOLD='\033[1m'
RESET='\033[0m'

# ---- Pre-flight state -------------------------------------------------------

PREFLIGHT_WARNINGS=0
PREFLIGHT_LOG=""

log_preflight() {
    local msg="$1"
    PREFLIGHT_LOG="${PREFLIGHT_LOG}${msg}\n"
}

# ---- Pre-flight functions ---------------------------------------------------

check_disk_space() {
    echo -e "${BOLD}Checking disk space...${RESET}"
    local avail_kb
    mkdir -p "${CHECKPOINT_BASE}"
    avail_kb=$(df --output=avail "${CHECKPOINT_BASE}" | tail -1 | tr -d ' ')
    local avail_gb=$((avail_kb / 1024 / 1024))

    if [ "$avail_gb" -lt "$MIN_DISK_GB" ]; then
        echo -e "  ${YELLOW}WARNING: Free disk space ${avail_gb} GB < ${MIN_DISK_GB} GB minimum${RESET}"
        PREFLIGHT_WARNINGS=$((PREFLIGHT_WARNINGS + 1))
        log_preflight "WARN: Disk space ${avail_gb} GB (need ${MIN_DISK_GB} GB)"
    else
        echo -e "  ${GREEN}OK: ${avail_gb} GB free (need ${MIN_DISK_GB} GB)${RESET}"
        log_preflight "OK: Disk space ${avail_gb} GB"
    fi
}

check_wsl_memory() {
    echo -e "${BOLD}Checking WSL2 memory...${RESET}"
    local mem_kb
    mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
    local mem_gb=$((mem_kb / 1024 / 1024))

    if [ "$mem_gb" -lt "$MIN_WSL_MEM_GB" ]; then
        echo -e "  ${YELLOW}WARNING: WSL2 memory ${mem_gb} GB < ${MIN_WSL_MEM_GB} GB minimum${RESET}"
        echo -e "  ${YELLOW}Consider updating .wslconfig: memory=${MIN_WSL_MEM_GB}GB${RESET}"
        PREFLIGHT_WARNINGS=$((PREFLIGHT_WARNINGS + 1))
        log_preflight "WARN: WSL2 memory ${mem_gb} GB (need ${MIN_WSL_MEM_GB} GB)"
    else
        echo -e "  ${GREEN}OK: ${mem_gb} GB available (need ${MIN_WSL_MEM_GB} GB)${RESET}"
        log_preflight "OK: WSL2 memory ${mem_gb} GB"
    fi
}

check_gpu_processes() {
    echo -e "${BOLD}Checking for competing GPU processes...${RESET}"
    local procs
    procs=$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory \
            --format=csv,noheader,nounits 2>/dev/null || true)

    if [ -n "$procs" ]; then
        echo -e "  ${YELLOW}WARNING: Competing GPU processes detected:${RESET}"
        echo "$procs" | while IFS= read -r line; do
            echo -e "    $line"
        done
        echo ""
        log_preflight "WARN: Competing GPU processes:\n${procs}"
        read -p "  Continue anyway? (y/N) " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 1
        fi
    else
        echo -e "  ${GREEN}OK: No competing GPU processes${RESET}"
        log_preflight "OK: No competing GPU processes"
    fi
}

write_preflight_log() {
    mkdir -p "$PREFLIGHT_LOG_DIR"
    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_path="${PREFLIGHT_LOG_DIR}/preflight_${timestamp}.log"
    echo -e "Pre-flight check results: $(date -u +"%Y-%m-%dT%H:%M:%SZ")\n" > "$log_path"
    echo -e "$PREFLIGHT_LOG" >> "$log_path"
    echo -e "\nWarnings: ${PREFLIGHT_WARNINGS}" >> "$log_path"
    echo -e "  Pre-flight log written to: ${log_path}"
}

# ---- Resume detection -------------------------------------------------------

RESUME_DIR=""

detect_existing_run() {
    echo -e "\n${BOLD}Checking for existing 1B runs...${RESET}"
    local latest_run
    latest_run=$(ls -td "${CHECKPOINT_BASE}"/1B_* 2>/dev/null | head -1 || true)

    if [ -n "$latest_run" ] && [ -f "$latest_run/latest" ]; then
        local latest_step
        latest_step=$(cat "$latest_run/latest")
        echo -e "  Found existing 1B run: ${BOLD}${latest_run}${RESET}"
        echo -e "  Latest checkpoint: ${latest_step}"
        echo ""
        read -p "  Resume this run? (Y/n) " -r
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            RESUME_DIR="$latest_run"
            echo -e "  ${GREEN}Will resume from ${latest_run}${RESET}"
        else
            echo -e "  Starting fresh run"
        fi
    else
        echo -e "  ${GREEN}No existing 1B runs found. Starting fresh.${RESET}"
    fi
}

# ---- tmux check -------------------------------------------------------------

check_tmux() {
    if [ -z "${TMUX:-}" ]; then
        echo -e "\n${YELLOW}NOTE: Not running inside tmux.${RESET}"
        echo -e "  This training run is projected to take ~45 hours."
        echo -e "  Recommend running inside tmux to survive terminal disconnects."
        echo ""
        echo -e "  Start a new session:    ${BOLD}tmux new-session -s ${TMUX_SESSION}${RESET}"
        echo -e "  Reconnect later:        ${BOLD}tmux attach -t ${TMUX_SESSION}${RESET}"
        echo ""
        read -p "  Continue without tmux? (y/N) " -r
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted. Start tmux first, then re-run this script."
            exit 0
        fi
    else
        echo -e "\n${GREEN}Running inside tmux session.${RESET}"
    fi
}

# ---- Command assembly -------------------------------------------------------

assemble_and_run() {
    echo -e "\n${BOLD}Assembling training command...${RESET}"

    local cmd="deepspeed model/train.py"
    cmd+=" --config ${CONFIG}"
    cmd+=" --data-dir ${DATA_DIR}"
    cmd+=" --deepspeed_config ${DS_CONFIG}"
    cmd+=" --checkpoint-interval ${CHECKPOINT_INTERVAL}"
    cmd+=" --checkpoint-base ${CHECKPOINT_BASE}"
    cmd+=" --max-epochs 3"

    if [ -n "$RESUME_DIR" ]; then
        cmd+=" --resume ${RESUME_DIR}"
    fi

    echo -e "\n${BOLD}Command:${RESET}"
    echo -e "  ${cmd}"
    echo ""

    echo -e "${BOLD}========================================${RESET}"
    echo -e "${BOLD}  Starting 1B training...${RESET}"
    echo -e "${BOLD}========================================${RESET}\n"

    # Replace this script's process with the training process
    exec $cmd
}

# ---- Main flow --------------------------------------------------------------

echo -e "\n${BOLD}========================================${RESET}"
echo -e "${BOLD}  ALS-LM 1B Production Training${RESET}"
echo -e "${BOLD}========================================${RESET}\n"

# Pre-flight checks
check_disk_space
check_wsl_memory
check_gpu_processes
write_preflight_log

# Warn on pre-flight issues
if [ "$PREFLIGHT_WARNINGS" -gt 0 ]; then
    echo ""
    echo -e "${YELLOW}Pre-flight warnings detected (${PREFLIGHT_WARNINGS} issue(s)).${RESET}"
    read -p "  Continue? (y/N) " -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Resume detection
detect_existing_run

# tmux check
check_tmux

# Launch training
assemble_and_run
