#!/bin/bash

set -e

echo "Starting Docker infrastructure..."
docker compose up -d

sleep 3

echo "Starting training..."
python -m morphseg.train "$@"

# Optional
# docker compose down