import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class Standardizer(object):
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        return self

    def transform(self, X):
        X_transformed = (X - self.mean) / self.std
        X_transformed[np.isinf(X_transformed)] = 0.0
        X_transformed[np.isnan(X_transformed)] = 0.0
        return X_transformed
