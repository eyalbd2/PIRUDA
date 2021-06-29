# PIRUDA

Code for "PIRUDA: Probing-based Information Removal for Unsupervised Domain Adaptation", a final-project performed during a seminar in natural language processing.

## Experimental Pipline
This lists the stages that need to be executed in order to reproduce our models and baselines results.

### 0. Initialize project

Clone git project.

```
git clone https://github.com/eyalbd2/PIRUDA
```

### 1. Setup a virtual env
Create and activate a dedicated conda environment.


### 2. Run experiments of (a) PIRUDA models and baselines.

##### For PIRUDA models, run the following command:

```
bash run_piruda_experiments.sh <GPU_ID> <TASK_NAME>
```

Where ''TASK_NAME'' can be on of the followings: [aspect, sentiment, mnli, rumour].

##### For baseline models, run the following command:
```
bash run_baselines_experiments.sh <GPU_ID> <MODEL_NAME> <DATA_NAME>
```

Where ''MODEL_NAME'' can be on of the followings: [f-bert, ft-bert, dann-bert, irm-bert], and ''DATA_NAME'' is in the following group: [absa, blitzer, mnli, rumour]

