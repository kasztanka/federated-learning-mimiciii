import math
import numpy as np
import torch


TOO_LITTLE_ELEMENTS_WARNING = 'Warning: Too little elements to split. Some splits might get 0 elements.'


def uniform(X, y, splits):
    if len(X) < splits:
        print(TOO_LITTLE_ELEMENTS_WARNING)
    split_size = math.floor(len(X) / splits)
    rest = len(X) % splits
    split_sections = [split_size + 1 if i < rest else split_size for i in range(splits)]
    split_X = torch.split(X, split_sections, dim=0)
    split_y = torch.split(y, split_sections, dim=0)

    for i in range(splits):
        yield split_X[i], split_y[i]


def linear(X, y, splits):
    split_sum = splits * (splits + 1) / 2
    smallest_split_size = math.floor(len(X) / split_sum)
    if smallest_split_size == 0:
        print(TOO_LITTLE_ELEMENTS_WARNING)

    split_sections = [smallest_split_size * (i + 1) for i in range(splits)]
    rest = len(X) - sum(split_sections)
    for i in range(rest):
        split_sections[i % len(split_sections)] += 1

    split_X = torch.split(X, split_sections, dim=0)
    split_y = torch.split(y, split_sections, dim=0)

    for i in range(splits):
        yield split_X[i], split_y[i]


def beta_distribution(X, y, splits, a, b):
    if len(X) < splits:
        print(TOO_LITTLE_ELEMENTS_WARNING)
        split_sections = [1 if i < len(X) else 0 for i in range(splits)]
    else:
        rng = np.random.default_rng()
        beta_draws = list(rng.beta(a, b, splits))
        draw_sum = sum(beta_draws)

        split_sections = [1 for _ in range(splits)]
        remaining = len(X) - splits
        for i in range(splits - 1):
            split_sections[i] = int(round(remaining / draw_sum * beta_draws[i]))
        rest = len(X) - sum(split_sections)
        split_sections[-1] += rest

    split_X = torch.split(X, split_sections, dim=0)
    split_y = torch.split(y, split_sections, dim=0)

    for i in range(splits):
        yield split_X[i], split_y[i]


def beta_right_skewed(X, y, splits):
    return beta_distribution(X, y, splits, a=2, b=5)


def beta_center(X, y, splits):
    return beta_distribution(X, y, splits, a=2, b=2)


def beta_left_skewed(X, y, splits):
    return beta_distribution(X, y, splits, a=5, b=2)
