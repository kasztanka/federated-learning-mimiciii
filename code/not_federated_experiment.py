import numpy as np
import torch
from torch.nn import BCELoss
from torch.optim import RMSprop
from torch.utils.data import DataLoader, TensorDataset

from utils import EarlyStopping


class NotFederatedExperiment:
    def __init__(self, config):
        self.config = config

    def create_datasets(self, X, y, train_idx, test_idx):
        tensor_X, tensor_y = torch.tensor(X).float(), torch.tensor(y).float()

        split = int(np.floor(self.config.validation_split * len(train_idx)))
        train_idx, valid_idx = train_idx[split:], train_idx[:split]
        indices = [train_idx, valid_idx, test_idx]

        loaders = []
        for idx in indices:
            dataset = TensorDataset(tensor_X[idx], tensor_y[idx])
            loaders.append(DataLoader(dataset, batch_size=self.config.batch_size))

        return loaders

    def train(self, model, train_loader, valid_loader):
        criterion = BCELoss()  # binary cross-entropy
        optimizer = RMSprop(model.parameters(), lr=self.config.learning_rate)
        early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)

        epochs_finished = 0
        for _ in range(self.config.epochs):

            model.train()
            for data, target in train_loader:
                optimizer.zero_grad()

                output = model(data)

                loss = criterion(output, target)
                loss.backward()

                optimizer.step()

            model.eval()
            valid_losses = []
            for data, target in valid_loader:
                output = model(data)
                loss = criterion(output, target)
                valid_losses.append(loss.item())
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
            end = start + len(target)
            targets[start:end] = target

            with torch.no_grad():
                predictions[start:end] = model(data)
            start = end
        return predictions, targets
