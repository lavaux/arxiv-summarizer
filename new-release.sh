#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 [--force] [-h]"
    echo "  --force   Force the release by replacing --dry-run with --no-ci"
    echo "  -h        Show this help message and exit"
    exit 0
}

# Initialize arguments
DRY_RUN="--noop"
FORCE_MODE=false

# Process arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Build the semantic-release command
SEMANTIC_RELEASE_CMD="semantic-release"
if [ "$FORCE_MODE" = true ]; then
    DRY_RUN=""
fi
SEMANTIC_RELEASE_CMD="$SEMANTIC_RELEASE_CMD $DRY_RUN -v version --vcs-release"

# Execute commands
if ! poetry build; then
    echo "Error: poetry build failed. Aborting release."
    exit 1
fi

echo "Executing: $SEMANTIC_RELEASE_CMD"
eval "$SEMANTIC_RELEASE_CMD"
