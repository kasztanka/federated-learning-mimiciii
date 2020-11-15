# Federated learning with MIMIC-III

The aim of this project was to experiment with [PySyft](https://github.com/OpenMined/PySyft), a federate learning library, on medical data.

As a reference point we used experiments on MIMIC-III dataset from "[Benchmarking Deep Learning Models on Large Healthcare Datasets](https://www.sciencedirect.com/science/article/pii/S1532046418300716)". The authors of the paper preprocessed the MIMIC-III dataset and created several feature sets that were then used in experiments. For our experiments, we used the one called Feature Set A, since it has the smallest number of features. What’s more, we used the version of the data in which the features were extracted for the first 24 hours after ICU admission.

In our experiments, we work on two out of three prediction problems stated in the paper: mortality and ICD-9 code group prediction. The first one is binary classification, the second one is multi-label classification. Both are predicted for the first 24 hours after ICU admission.

As an ML model we used a feed-forward neural network having the same architecture as the one in the paper. It’s used for both classification problems. The only difference is the number of outputs of the network: one for mortality, and twenty for ICD-9 code groups. In general, there are far more than twenty ICD-9 codes, but the benchmark’s authors grouped them into twenty diagnosis groups.

## Requirements

First, to download the project run:
```
git clone https://github.com/kasztanka/federated-learning-mimiciii.git
cd federated-learning-mimiciii
```

To recreate our experiments, a folder `data` needs to be created in the main folder of this repository and the following files need to be in it:

- 5-folds.npz - a file with folds
- imputed-normed-ep_1_24.npz - a file with labels
- input.csv - a file with features

A way to obtain these files is described in [this repository](https://github.com/USC-Melady/Benchmarking_DL_MIMICIII).

## Configuring the experiments

The experiments can be configured in the `code/experiments_configuration.json` file. This file may contain multiple experiment setups. Each setup is an item in the `experiments` list. In particular, we may provide a list of values for:
 - `train_size` - the number of examples in a train set,
 - `num_of_workers` - number of nodes,
 - `node_distribution` - node size distribution.

The `train_size` and `num_of_workers` parameters should be positive integers. Additionally, `train_size` can be set to `null` value, which stands for the maximum number of training examples. The `node_distribution` parameter should be one of: `"uniform"`, `"beta_center"`, `"beta_left_skewed"`, `"beta_right_skewed"`, `"linear"`.

Default values for all the parameters are set in the `default` field. In our experiments the default values were: using all available examples as training set, 32 grid nodes, and a uniform node size distribution.

The configuration file should also contain a value for `nodes_type` field in the default settings. It may be on of the following:
- `no_nodes` - for the non-federated experiments,
- `virtual` - for the federated experiments with PySyft's virtual workers,
- `real` - for the federated experiments with PySyft's `DataCentricFLClient` workers.

The configuration in the JSON file does not allow to set the classification problem. This setting must be specified in a Dockerfile and it can be set to `mortality` for mortality prediction, or to `icd9` for multi-label ICD-9 code group problem.

Only one problem and only one node type can be tested at a time.

## Running the experiments

### Federated with real nodes

To run the experiments use the following commands:
```
python create_docker_compose_for_experiments.py
docker-compose up --build
```

This will create a docker-compose file and run it. By default the results will be saved in `./results/`.

Additionally, you can pass a maximal number of workers used in experiments as a parameter to the Python script:
```
python create_docker_compose.py 32
```
The docker-compose.experiments.yml file in this repository is a sample file created by this script for three workers.

NOTE: If you get an error about no associated hostname, verify if a sufficient number of workers in the experiment is started by the docker-compose file.

### Others

To run the not federated experiments or federated ones with virtual workers, change the configuration and run:
```
docker build --tag federated_experiments .
docker run -v /absolute/path/to/results:/results federated_experiments
```

## Results

The results of our experiments can be found in the `analyze_results.ipynb` notebook.