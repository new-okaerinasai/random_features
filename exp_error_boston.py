from random_features import RandomFeatures
from sklearn.metrics.pairwise import rbf_kernel, cosine_distances
from sklearn.metrics import mean_squared_error as mse
from sklearn.datasets import load_boston, load_iris
import numpy as np
import matplotlib.pyplot as plt
N_SAMPLES, DIM = 1000, 200


if __name__ == "__main__":
    digits = load_boston()
    X, y = digits.data, digits.target.astype(np.int) - 1
    # X = np.random.randn(1000, 200)
    errs_gauss = []
    errs_angle = []
    for n in range(1, 5000, 200):
        gauss = RandomFeatures(gamma=0.1, n=n, metric="rbf")
        angle = RandomFeatures(gamma=2, n=n, metric="angle")

        G_gauss_pred = gauss.fit(X).compute_kernel(X)
        G_gauss = rbf_kernel(X)

        angle = RandomFeatures(gamma=2, n=n, metric="angle")

        G_angle_pred = angle.fit(X).compute_kernel(X)
        G_angle = rbf_kernel(X)
        errs_gauss.append(mse(G_gauss, G_gauss_pred))
        errs_angle.append(mse(G_angle, G_angle_pred))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    for ax, data, title in zip(axes, [errs_gauss, errs_angle], ["Gaussian", "Angle"]):
        ax.plot(np.arange(0, 5000, 200), data)
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("MSE")
        ax.set_title(title)
        ax.set_yscale("log")
    fig.suptitle("Boston Housing")
    plt.savefig("gauss_usps_err.png")
    plt.show()
