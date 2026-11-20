#!/usr/bin/env bash
# Copyright (c) OTAVSR. All rights reserved.
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun --nproc_per_node=$GPUS --master_port=$PORT \
    "$(dirname "$0")/test.py" "$CONFIG" --checkpoint "$CHECKPOINT" "${@:4}"
