#!/bin/bash

echo "Starting Docker infrastructure..."
docker compose up -d

sleep 3

echo "Starting training..."
python src/train.py "$@"

# Optional
# docker compose down