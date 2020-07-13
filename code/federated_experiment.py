import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import RMSprop

import syft as sy
from syft.federated.floptimizer import Optims

from utils import EarlyStopping


class FederatedExperiment:
    def __init__(self, experiment_id, hook, model_config, num_of_workers, node_distribution):
        self.experiment_id = experiment_id
        self.hook = hook
        self.model_config = model_config
        self.num_of_workers = num_of_workers
        self.node_distribution = node_distribution

    def create_workers(self):
        worker_ids = [str(self.experiment_id) + "-" + str(i) for i in range(self.num_of_workers)]
        return [sy.VirtualWorker(self.hook, id=worker_id) for worker_id in worker_ids]

    def distribute_dataset(self, X, y, train_idx, test_idx, workers):
        tensor_X, tensor_y = torch.tensor(X).float(), torch.tensor(y).float()

        num_train = len(train_idx)
        split = int(np.floor(self.model_config.validation_split * num_train))
        train_idx, valid_idx = train_idx[split:], train_idx[:split]
        indices = [train_idx, valid_idx, test_idx]
        tags = ['train', 'valid', 'test']

        for idx, tag in zip(indices, tags):
            for i, (part_x, part_y) in enumerate(self.node_distribution(tensor_X[idx], tensor_y[idx], len(workers))):
                tag_X = part_x.tag("#X", f"#{tag}").describe("")
                tag_y = part_y.tag("#Y", f"#{tag}").describe("")

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
        for _ in range(self.model_config.epochs):

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
    def predict(model, data_loader, output_size):
        model.eval()

        num_elements = sum([len(data) for data, _ in data_loader])

        predictions = torch.zeros(num_elements, output_size)
        targets = torch.zeros(num_elements, output_size)

        start = 0
        for data, target in data_loader:

            target = target.get()
            end = start + len(target)

            targets[start:end] = target

            model.send(data.location)
            with torch.no_grad():
                output = model(data)
                predictions[start:end] = output.get()
            model.get()
            start = end
        return predictions, targets
