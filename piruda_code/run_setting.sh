#!/bin/bash
# Run this command : bash run_setting.sh <GPU_ID> <TASK_NAME> <TRG_DOMAIN>

TASK_NAME=$1
TRG_DOMAIN=$2

echo "task $TASK_NAME domain $TRG_DOMAIN"
CUDA_VISIBLE_DEVICES=$1 python -u ./piruda_code/fineTuning.py -task "$TASK_NAME" -domain "$TRG_DOMAIN"
CUDA_VISIBLE_DEVICES=$1 python -u ./piruda_code/ftDataParsing.py -task "$TASK_NAME" -domain "$TRG_DOMAIN"
CUDA_VISIBLE_DEVICES=$1 python -u ./piruda_code/INLP_remove.py -task "$TASK_NAME" -domain "$TRG_DOMAIN" -iter 100
CUDA_VISIBLE_DEVICES=$1 python -u ./piruda_code/INLP_predict.py -task "$TASK_NAME" -domain "$TRG_DOMAIN" -iter 100
CUDA_VISIBLE_DEVICES=$1 python -u ./piruda_code/PDA_remove.py -task "$TASK_NAME" -domain "$TRG_DOMAIN"
CUDA_VISIBLE_DEVICES=$1 python -u ./piruda_code/PDA_predict.py -task "$TASK_NAME" -domain "$TRG_DOMAIN"
