import csv
import json
import os
import sys
import time
from multiprocessing import Process, Queue

import numpy as np
import torch
from sklearn import metrics

import syft as sy

from distributions import beta_center, beta_left_skewed, beta_right_skewed, linear, uniform
from federated_experiment import FederatedExperiment
from not_federated_experiment import NotFederatedExperiment
from utils import build_model, Config, ICD9_SETUP, Metric, MORTALITY_SETUP, Standardizer


def load_data(experiment_setup):
    data_folder = os.getenv('DATA')
    data_filename = os.path.join(data_folder, 'imputed-normed-ep_1_24.npz')
    folds_filename = os.path.join(data_folder, '5-folds.npz')
    features_filename = os.path.join(data_folder, 'input.csv')

    if not os.path.exists(data_folder):
        print(f'Wrong data_folder specified. This folder must exist')
        exit(1)

    folds_file = np.load(folds_filename, allow_pickle=True)
    folds = folds_file[experiment_setup.folds_file][0][0]

    data_file = np.load(data_filename, allow_pickle=True)
    y = data_file[experiment_setup.y_label][:, :experiment_setup.output_size]
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


def gather_results(time_measurements, output_size, metric_list, y_true, y_soft, y_pred):
    results = []
    for i in range(output_size):
        single_output_results = dict(time_measurements)
        single_output_results['task'] = i + 1
        for metric in metric_list:
            if metric.use_soft:
                score = metric.function(y_true[:, i], y_soft[:, i])
            else:
                score = metric.function(y_true[:, i], y_pred[:, i])
            single_output_results[metric.name] = score
        results.append(single_output_results)
    return results


def run_experiment(queue, not_federated, experiment, model, X, y, train_idx, test_idx, metric_list, output_size):
    time_measurements = {}

    standardizer = Standardizer()
    standardizer.fit(X[train_idx])
    X_transformed = standardizer.transform(X)

    if not_federated:
        train_loader, valid_loader, test_loader = experiment.create_datasets(X_transformed, y, train_idx, test_idx)
        time_measurements['collecting_datasets'] = 0

        start = time.time()
        model, finished_epochs = experiment.train(model, train_loader, valid_loader)
        training_time = time.time() - start
    else:
        workers = experiment.create_workers()
        grid = experiment.distribute_dataset(X_transformed, y, train_idx, test_idx, workers)

        start = time.time()
        train_loader, valid_loader, test_loader = experiment.collect_datasets(grid)
        time_measurements['collecting_datasets'] = time.time() - start

        start = time.time()
        model, finished_epochs = experiment.train(model, train_loader, valid_loader, workers)
        training_time = time.time() - start
    time_measurements['training'] = training_time
    time_measurements['training_per_epoch'] = training_time / finished_epochs

    start = time.time()
    y_soft, y_true = experiment.predict(model, test_loader, output_size)
    y_pred = (y_soft > 0.5).type(torch.int)
    time_measurements['prediction'] = time.time() - start

    results = gather_results(time_measurements, output_size, metric_list, y_true, y_soft, y_pred)

    if not not_federated:
        experiment.clean_up(workers)
        del workers, grid
    del train_loader, valid_loader, test_loader
    queue.put(results)


def main():
    if len(sys.argv) != 3:
        raise ValueError('Please specify two arguments: problem name and experiments configuration file')

    allowed_experiments = {
        'icd9': ICD9_SETUP,
        'mortality': MORTALITY_SETUP
    }
    experiment_setup = allowed_experiments.get(sys.argv[1], None)
    if experiment_setup is None:
        raise ValueError(f'Wrong problem name. Allowed values are: {list(allowed_experiments.keys())}')

    configuration_filename = os.path.join(os.getenv('CODE'), sys.argv[2])
    if not os.path.exists(configuration_filename):
        raise ValueError('Specified experiments configuration file does not exist.')

    with open(configuration_filename) as f:
        config_as_json = json.load(f)
    configurations = denormalize_config(config_as_json)

    model_config = Config()
    hook = sy.TorchHook(torch)

    X, y, folds = load_data(experiment_setup)

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
    results_filename = os.path.join(results_folder, experiment_setup.results_filename + '.csv')

    fieldnames = list(configurations[0].keys())
    fieldnames += [metric.name for metric in metric_list]
    fieldnames += ['collecting_datasets', 'training', 'training_per_epoch', 'prediction', 'task']
    with open(results_filename, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    node_distribution_str2func = {
        'beta_center': beta_center,
        'beta_right_skewed': beta_right_skewed,
        'beta_left_skewed': beta_left_skewed,
        'linear': linear,
        'uniform': uniform,
    }

    repetitions = len(folds)
    for i, experiment_config in enumerate(configurations):
        for j, (train_idx, valid_idx, test_idx) in enumerate(folds):
            print(f'config: {i+1}/{len(configurations)} repetition: {j+1}/{repetitions}', flush=True)
            print(experiment_config, flush=True)

            not_federated = experiment_config['nodes_type'] == 'no_nodes'
            if not_federated:
                experiment = NotFederatedExperiment(model_config)
            elif experiment_config['nodes_type'] == 'real':
                experiment = FederatedExperiment(
                    hook, model_config, experiment_config['num_of_workers'],
                    node_distribution_str2func[experiment_config['node_distribution']]
                )
            else:
                raise ValueError(f'Wrong nodes type. Allowed values are: "no nodes", "real" or "virtual"')

            model = build_model(model_config, n_features=X.shape[1], output_size=experiment_setup.output_size)

            train_idx = np.concatenate((train_idx, valid_idx))
            if experiment_config['train_size'] is not None:
                train_idx = train_idx[:experiment_config['train_size']]

            queue = Queue()
            p = Process(
                target=run_experiment,
                args=(
                    queue, not_federated, experiment,
                    model, X, y, train_idx, test_idx,
                    metric_list, experiment_setup.output_size)
            )
            p.start()
            p.join()
            results = queue.get()

            for name, value in experiment_config.items():
                for k in range(experiment_setup.output_size):
                    results[k][name] = value

            with open(results_filename, mode='a') as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                for k in range(experiment_setup.output_size):
                    writer.writerow(results[k])

            del experiment, p, queue


if __name__ == "__main__":
    main()
