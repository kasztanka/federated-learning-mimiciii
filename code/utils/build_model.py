import torch.nn as nn


def build_model(config, n_features):
    layers = []

    hidden_dim = config.hidden_dim
    if hidden_dim is None:
        hidden_dim = 8 * n_features

    layers.append(nn.Linear(n_features, hidden_dim))
    layers.append(nn.Sigmoid())
    layers.append(nn.Dropout(0.1))

    for t in range(config.ffn_depth - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Sigmoid())
        layers.append(nn.Dropout(0.2))

    layers.append(nn.Linear(hidden_dim, 1))

    if config.batch_normalization:
        layers.append(nn.BatchNorm1d(1))

    layers.append(nn.Sigmoid())

    return nn.Sequential(*layers)
