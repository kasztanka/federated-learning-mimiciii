import math
import string

import numpy as np
import syft as sy
import torch
from syft.federated.floptimizer import Optims
from torch.nn import BCELoss
from tqdm.notebook import tqdm

from utils import EarlyStopping


class FederatedExperiment:
    def __init__(self, experiment_id, hook, model_config, num_of_workers):
        self.experiment_id = experiment_id
        self.hook = hook
        self.model_config = model_config
        self.num_of_workers = num_of_workers

    def create_workers(self):
        worker_ids = [string.ascii_letters[i] + str(self.experiment_id) for i in range(self.num_of_workers)]
        return [sy.VirtualWorker(self.hook, id=worker_id) for worker_id in worker_ids]

    def distribute_dataset(self, X, y, train_idx, test_idx, workers):
        tensor_X, tensor_y = torch.tensor(X), torch.tensor(y).view(-1, 1)

        num_train = len(train_idx)
        split = int(np.floor(self.model_config.validation_split * num_train))
        train_idx, valid_idx = train_idx[split:], train_idx[:split]
        indices = [train_idx, valid_idx, test_idx]
        tags = ['train', 'valid', 'test']

        for idx, tag in zip(indices, tags):
            split_per_worker = math.ceil(len(tensor_X[idx]) / len(workers))
            split_X = torch.split(tensor_X[idx], split_per_worker, dim=0)
            split_y = torch.split(tensor_y[idx], split_per_worker, dim=0)

            for i in range(len(workers)):
                tag_X = split_X[i].tag("#X", f"#{tag}").describe("")
                tag_y = split_y[i].tag("#Y", f"#{tag}").describe("")

                tag_X.send(workers[i], garbage_collect_data=False)
                tag_y.send(workers[i], garbage_collect_data=False)

        return sy.PrivateGridNetwork(*workers)

    def collect_datasets(self, grid):
        loaders = []
        tags = ['train', 'valid', 'test']
        for tag in tags:
            found_X = grid.search("#X", f"#{tag}")
            found_y = grid.search("#Y", f"#{tag}")

            datasets = []
            for worker in found_X.keys():
                datasets.append(sy.BaseDataset(found_X[worker][0], found_y[worker][0]))

            dataset = sy.FederatedDataset(datasets)
            loaders.append(sy.FederatedDataLoader(dataset, batch_size=self.model_config.batch_size))

        return loaders

    def train(self, model, train_loader, valid_loader, workers):
        criterion = BCELoss()  # binary cross-entropy
        # for RMSprop in PySyft each worker needs its own optimizer
        worker_ids = [worker.id for worker in workers]
        optims = Optims(worker_ids, optim=RMSprop(model.parameters(), lr=self.model_config.learning_rate))
        early_stopping = EarlyStopping(patience=self.model_config.early_stopping_patience)

        epochs_finished = 0
        for _ in tqdm(range(self.model_config.epochs)):

            model.train()
            for data, target in train_loader:
                model.send(data.location)

                opt = optims.get_optim(data.location.id)
                opt.zero_grad()

                output = model(data)

                loss = criterion(output, target)
                loss.backward()

                opt.step()
                model.get()

            model.eval()
            valid_losses = []
            for data, target in valid_loader:
                model.send(data.location)

                output = model(data)
                loss = criterion(output, target)
                valid_losses.append(loss.get().item())

                model.get()
            valid_loss = np.average(valid_losses)

            epochs_finished += 1

            if early_stopping.should_early_stop(valid_loss, model):
                break

        model.load_state_dict(early_stopping.best_model_state)

        return model, epochs_finished

    @staticmethod
    def predict(model, data_loader):
        model.eval()

        num_elements = sum([len(data) for data, _ in data_loader])

        predictions = torch.zeros(num_elements)
        targets = torch.zeros(num_elements)

        start = 0
        for data, target in data_loader:

            target = target.get().view(-1)
            end = start + len(target)

            targets[start:end] = target

            model.send(data.location)
            with torch.no_grad():
                output = model(data)
                predictions[start:end] = output.get().view(-1)
            model.get()
            start = end
        return predictions, targets
