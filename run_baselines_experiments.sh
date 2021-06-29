#!/bin/bash
# Run this command : bash run_baselines_experiments.sh <GPU_ID> <MODEL_NAME> <DATA_NAME>

MODEL_NAME=$2 # (f-bert, ft-bert, dann-bert, irm-bert)
DATA_NAME=$3 # (absa, blitzer, mnli, rumour)

if [ "$MODEL_NAME" == "f-bert" ] ; then
  N_EPOCHS=5
  BATCH_SIZE=50
  MODEL_NAME=f-bert
  train_file_name=ft-bert
elif [ "$MODEL_NAME" == "ft-bert" ] ; then
  N_EPOCHS=5
  BATCH_SIZE=50
  MODEL_NAME=ft-bert
  train_file_name=ft-bert
elif [ "$MODEL_NAME" == "dann-bert" ] ; then
  N_EPOCHS=5
  BATCH_SIZE=50
  MODEL_NAME=dann-bert
  train_file_name=dann-bert
elif [ "$MODEL_NAME" == "irm-bert" ] ; then
  N_EPOCHS=5
  BATCH_SIZE=13
  MODEL_NAME=irm-bert
  train_file_name=irm-bert
else
  echo "Invalid model name: $MODEL_NAME (f-bert, ft-bert, dann-bert, irm-bert)"
  exit
fi

if [ "$DATA_NAME" == "absa" ] ; then
  DATA_TYPE=absa
  DOMAINS=(device laptops rest service)
elif [ "$DATA_NAME" == "blitzer" ] ; then
  DATA_TYPE=blitzer
  DOMAINS=(airline books dvd electronics kitchen)
elif [ "$DATA_NAME" == "mnli" ] ; then
  DATA_TYPE=mnli
  DOMAINS=(fiction government slate telephone travel)
elif [ "$DATA_NAME" == "rumour" ] ; then
  DATA_TYPE=rumour
  DOMAINS=(charliehebdo ferguson germanwings-crash ottawashooting sydneysiege)
else
  echo "Invalid data name: $DATA_NAME (absa, blitzer, mnli, rumour)"
  exit
fi

for i in "${!DOMAINS[@]}"
do
  TRG_DOMAIN=${DOMAINS[$i]}
  DOMAINS_TMP=("${DOMAINS[@]}")
  unset DOMAINS_TMP[$i]
  SRC_DOMAINS=$(echo ${DOMAINS_TMP[*]} | tr ' ' ',')
  MODELS_DIR=baseline-models/${DATA_TYPE}/${MODEL_NAME}/${TRG_DOMAIN}/

  echo "Running $MODEL_NAME experiment for $SRC_DOMAINS as source and $TRG_DOMAIN as target domains on GPU $1"

  set -x
  if [ "$MODEL_NAME" == "f-bert" ] ; then
    CUDA_VISIBLE_DEVICES=$1 python ./baselines_code/train_${train_file_name}.py \
    --src_domains=${SRC_DOMAINS} \
    --trg_domains=${TRG_DOMAIN} \
    --data_type=${DATA_TYPE} \
    --do_train \
    --output_dir=${MODELS_DIR} \
    --learning_rate=5e-5 \
    --num_train_epochs=${N_EPOCHS} \
    --train_batch_size=${BATCH_SIZE} \
    --save_according_to=acc \
    --freeze_bert
  else
    CUDA_VISIBLE_DEVICES=$1 python ./baselines_code/train_${train_file_name}.py \
    --src_domains=${SRC_DOMAINS} \
    --trg_domains=${TRG_DOMAIN} \
    --data_type=${DATA_TYPE} \
    --do_train \
    --output_dir=${MODELS_DIR} \
    --learning_rate=5e-5 \
    --num_train_epochs=${N_EPOCHS} \
    --train_batch_size=${BATCH_SIZE} \
    --save_according_to=acc
  fi
done