#!/usr/bin/env bash

set -x
set -e

TASK="wiki5m_trans"

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

WNAME=$1
echo "wname: ${WNAME}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/${TASK}_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${TASK}"
fi

model_path="bert"
task="wiki5m_trans"
DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data/${task}"
fi

if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    model_path=$1
    shift
fi

neighbor_weight=0.05

python train_and_eval.py \
--model-dir "${OUTPUT_DIR}" \
--pretrained-model bert-base-uncased \
--pooling mean \
--lr 3e-5 \
--train-path "${DATA_DIR}/train.txt.json" \
--valid-path "${DATA_DIR}/valid.txt.json" \
--test-path "${DATA_DIR}/test.txt.json" \
--task ${TASK} \
--batch-size 1024 \
--print-freq 20 \
--additive-margin 0.02 \
--use-amp \
--use-self-negative \
--finetune-t \
--pre-batch 0 \
--epochs 1 \
--workers 3 \
--max-to-keep 10 \
-wname $WNAME \
-uw \
--eval-model-path "${model_path}" \
--neighbor-weight "${neighbor_weight}" \
--project_name "wiki5m_trans" \
-sclw 1 
