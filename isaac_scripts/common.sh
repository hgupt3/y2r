#!/bin/bash
# ==============================================================================
# Y2R Common Script Helpers
# ==============================================================================
# Source this file in other scripts: source "$(dirname "$0")/common.sh"
# ==============================================================================

# Dynamically determine paths relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ISAACLAB_DIR="$REPO_ROOT/IsaacLab"

TASK="Isaac-Trajectory-Kuka-Allegro-v0"

# Find the latest checkpoint in a given log directory
# Usage: CHECKPOINT=$(find_latest "trajectory")
find_latest() {
    local dir="$1"
    find "$ISAACLAB_DIR/logs/rl_games/$dir" -name "*.pth" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

# Require a file to exist, exit with error if not found
# Usage: require_file "$CHECKPOINT"
require_file() {
    local file="$1"
    if [ ! -f "$file" ]; then
        echo "Error: File not found: $file"
        exit 1
    fi
}

# Parse common checkpoint arguments
# Sets: CHECKPOINT variable
# Usage: parse_checkpoint_args "trajectory" "$@"
#        shift $PARSED_ARGS
parse_checkpoint_args() {
    local log_dir="$1"
    shift
    CHECKPOINT=""
    PARSED_ARGS=0
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --continue)
                CHECKPOINT=$(find_latest "$log_dir")
                if [ -z "$CHECKPOINT" ]; then
                    echo "Error: No checkpoint found in logs/rl_games/$log_dir/"
                    exit 1
                fi
                echo "========================================"
                echo "Resuming from: $CHECKPOINT"
                echo "========================================"
                PARSED_ARGS=$((PARSED_ARGS + 1))
                shift
                ;;
            --checkpoint)
                CHECKPOINT="$2"
                require_file "$CHECKPOINT"
                echo "========================================"
                echo "Using checkpoint: $CHECKPOINT"
                echo "========================================"
                PARSED_ARGS=$((PARSED_ARGS + 2))
                shift 2
                ;;
            *)
                break
                ;;
        esac
    done
}

