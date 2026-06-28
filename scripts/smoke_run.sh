#!/bin/bash

set -e 

RUN_DIR="outputs/$(date +%Y-%m-%d/%H-%M-%S)"

uv run do-train \
    --config-name smoke_train \
    run.dir=$RUN_DIR/train

uv run run-test \
    --config-name smoke_test \
    run.dir=$RUN_DIR/test \
    run_info_dir=$RUN_DIR

echo "Smoke run completed"