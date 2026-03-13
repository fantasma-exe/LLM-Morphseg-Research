#!/bin/bash

set -e 

uv run python -m morphseg.train --config-name debug_train

echo "---- Train done ----"

uv run python -m morphseg.predict --config-name debug_predict

echo "--- Debug done ----" 