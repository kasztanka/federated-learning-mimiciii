import csv
import gc
import json
import os
import time

import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

import syft as sy

from federated_experiment import FederatedExperiment
from utils import build_model, Config, Metric, Standardizer


def load_data(config):
    data_folder = os.getenv('DATA')
    data_filename = os.path.join(data_folder, 'imputed-normed-ep_1_24.npz')
    folds_filename = os.path.join(data_folder, '5-folds.npz')
    features_filename = os.path.join(data_folder, 'input.csv')

    if not os.path.exists(data_folder):
        print(f'Wrong data_folder specified. This folder must exist')
        exit(1)

    folds_file = np.load(folds_filename, allow_pickle=True)
    folds = folds_file['folds_ep_mor'][config.label_type][0]

    data_file = np.load(data_filename, allow_pickle=True)
    y = data_file['adm_labels_all'][:, config.label_type]
    y = (y > 0).astype(float)

    X = np.genfromtxt(features_filename, delimiter=',')
    return X, y, folds


def denormalize_config(config_as_json):
    configurations = []
    default_values = config_as_json['default']
    for experiment in config_as_json['experiments']:
        name = experiment['parameters']['name']
        for value in experiment['parameters']['values']:
            config = {'experiment_type': experiment['type']}
            for d_name, d_value in default_values.items():
                if d_name == name:
                    config[name] = value
                else:
                    config[d_name] = d_value
            configurations.append(config)
    return configurations


def run_experiment(experiment, model, X, y, train_idx, test_idx, metric_list):
    results = {}

    standardizer = Standardizer()
    standardizer.fit(X[train_idx])
    X_transformed = standardizer.transform(X)

    workers = experiment.create_workers()

    grid = experiment.distribute_dataset(X_transformed, y, train_idx, test_idx, workers)

    start = time.time()
    train_loader, valid_loader, test_loader = experiment.collect_datasets(grid)
    results['collecting_datasets'] = time.time() - start

    start = time.time()
    model, finished_epochs = experiment.train(model, train_loader, valid_loader, workers)
    training_time = time.time() - start
    results['training'] = training_time
    results['training_per_epoch'] = training_time / finished_epochs

    start = time.time()
    y_soft, y_true = experiment.predict(model, test_loader)
    y_pred = (y_soft > 0.5).type(torch.int)
    results['prediction'] = time.time() - start

    for metric in metric_list:
        if metric.use_soft:
            score = metric.function(y_true, y_soft)
        else:
            score = metric.function(y_true, y_pred)
        results[metric.name] = score

    del train_loader, valid_loader, test_loader, workers, grid
    gc.collect()
    return results


def main():
    configuration_filename = os.path.join(os.getenv('CODE'), 'experiments_configuration.json')
    with open(configuration_filename) as f:
        config_as_json = json.load(f)
    configurations = denormalize_config(config_as_json)

    model_config = Config()
    hook = sy.TorchHook(torch)

    X, y, folds = load_data(model_config)

    metric_list = [
        Metric('accuracy', metrics.accuracy_score, use_soft=False),
        Metric('precision', metrics.precision_score, use_soft=False),
        Metric('recall', metrics.recall_score, use_soft=False),
        Metric('f1_score', metrics.f1_score, use_soft=False),
        Metric('roc_auc', metrics.roc_auc_score, use_soft=True),
        Metric('average_precision', metrics.average_precision_score, use_soft=True),
    ]

    results_folder = os.getenv('RESULTS')

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    results_filename = os.path.join(results_folder, 'experiment_results.csv')

    fieldnames = list(configurations[0].keys())
    fieldnames += [metric.name for metric in metric_list]
    fieldnames += ['collecting_datasets', 'training', 'training_per_epoch', 'prediction']
    with open(results_filename, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    repetitions = len(folds)
    progress_bar = tqdm(enumerate(configurations), total=len(configurations))
    for i, experiment_config in progress_bar:
        for j, (train_idx, valid_idx, test_idx) in enumerate(folds):
            progress_bar.set_postfix(config=str(experiment_config), iter=f'{j+1}/{repetitions}', refresh=True)

            experiment_id = i * repetitions + j
            experiment = FederatedExperiment(experiment_id, hook, model_config, experiment_config['num_of_workers'])

            model = build_model(model_config, n_features=X.shape[1])

            train_idx = np.concatenate((train_idx, valid_idx))
            if experiment_config['train_size'] is not None:
                train_idx = train_idx[:experiment_config['train_size']]

            results = run_experiment(experiment, model, X, y, train_idx, test_idx, metric_list)

            for name, value in experiment_config.items():
                results[name] = value

            with open(results_filename, mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writerow(results)


if __name__ == "__main__":
    main()
