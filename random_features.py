from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
import numpy as np
N_SAMPLES, DIM = 1000, 200


class RandomFeatures(BaseEstimator):
    def __init__(self, gamma=1, n=50, metric="rbf"):
        self.gamma = gamma
        self.metric = metric
        # Dimensionality D (number of MonteCarlo samples)
        self.n = n
        self.fitted = False

    def fit(self, X, y=None):
        """ Generates MonteCarlo random samples """
        d = X.shape[1]
        # Generate D iid samples from p(w)
        if self.metric == "rbf":
            self.w = np.sqrt(2 * self.gamma) * np.random.normal(size=(self.n, d))
        elif self.metric == "angle":
            self.w = np.random.randn(self.n, d)
            self.w /= np.linalg.norm(self.w, axis=1)[:, None]

        # Generate D iid samples from Uniform(0,2*pi)
        self.u = 2 * np.pi * np.random.rand(self.n)
        self.fitted = True
        return self

    def transform(self, X):
        """ Transforms the data X (n_samples, n_features) to the new map space Z(X) (n_samples, n_components)"""
        if not self.fitted:
            raise NotFittedError(
                "RBF_MonteCarlo must be fitted beform computing the feature map Z"
            )
        # Compute feature map Z(x):
        if self.metric == "angle":
            Z = np.sign(X @ self.w.T)
        else:
            Z = np.sqrt(2) * np.cos((X.dot(self.w.T) + self.u[None, :]))
        return Z

    def compute_kernel(self, X):
        """ Computes the approximated kernel matrix K """
        if not self.fitted:
            raise NotFittedError(
                "RBF_MonteCarlo must be fitted beform computing the kernel matrix"
            )
        Z = self.transform(X)
        K = Z @ Z.T / self.n
        return K
