#!/bin/bash
# Run this command : bash run_piruda_experiments.sh <GPU_ID> <TASK_NAME>

TASK_NAME=$2 # (aspect, sentiment, mnli, rumour)

if [ "$DATA_NAME" == "aspect" ] ; then
  DATA_TYPE=absa
  DOMAINS=(device laptops rest service)
elif [ "$DATA_NAME" == "sentiment" ] ; then
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
	./piruda_code/run_setting.sh $1 ${TASK_NAME} ${TRG_DOMAIN}
done
