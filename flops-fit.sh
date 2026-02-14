#!/usr/bin/env bash
#
# flops-fit: Scaling Laws Toolkit
#
# Usage:
#   flops-fit plan [args]       Generate sweep configurations
#   flops-fit train [args]      Run training experiments
#   flops-fit analyze [args]    Fit power laws to results
#   flops-fit visualize [args]  Generate plots
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    cat << EOF
flops-fit: Find compute-optimal models using IsoFLOPs scaling laws

Usage:
    flops-fit <command> [options]

Commands:
    plan        Generate sweep configurations
    train       Run training experiments  
    analyze     Fit power laws to results
    visualize   Generate plots

Quick start:
    flops-fit plan --config-name=presets/cpu_fast
    flops-fit train mode=mock
    flops-fit analyze
    flops-fit visualize

For command-specific help:
    ff-plan --help
    ff-train --help
    ff-analyze --help
    ff-visualize --help

EOF
}

if [[ $# -eq 0 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_help
    exit 0
fi

CMD="$1"
shift

case "$CMD" in
    plan)
        exec ff-plan "$@"
        ;;
    train)
        exec ff-train "$@"
        ;;
    analyze)
        exec ff-analyze "$@"
        ;;
    visualize|viz)
        exec ff-visualize "$@"
        ;;
    *)
        echo "Error: Unknown command '$CMD'"
        echo "Run 'flops-fit --help' for usage."
        exit 1
        ;;
esac
