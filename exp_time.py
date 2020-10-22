from random_features import RandomFeatures
from sklearn.metrics.pairwise import rbf_kernel, cosine_distances
from sklearn.metrics import mean_squared_error as mse
from sklearn.datasets import load_boston, load_iris
import numpy as np
import matplotlib.pyplot as plt
import time

N_SAMPLES, DIM = 1000, 200


if __name__ == "__main__":
    digits = load_boston()
    X, y = digits.data, digits.target.astype(np.int) - 1
    # X = np.random.randn(1000, 200)
    errs_gauss = []
    errs_angle = []
    total_time_gauss = []
    total_time_gauss_gt = []

    total_time_angle = []
    total_time_angle_gt = []
    for n in range(1, 5000, 200):
        time_gauss = []
        for restart in range(5):
            begin = time.time()
            gauss = RandomFeatures(gamma=0.1, n=n, metric="rbf")
            G_gauss_pred = gauss.fit(X).compute_kernel(X)
            time_gauss.append(time.time() - begin)
        total_time_gauss.append(np.mean(time_gauss))
        time_gt = []
        for restart in range(5):
            begin = time.time()
            G_gauss = rbf_kernel(X)
            time_gt.append(time.time() - begin)
        total_time_gauss_gt.append(np.mean(time_gt))

        time_angle = []
        for restart in range(5):
            begin = time.time()
            angle = RandomFeatures(gamma=2, n=n, metric="angle")
            G_angle_pred = angle.fit(X).compute_kernel(X)
            time_angle.append(time.time() - begin)
        total_time_angle.append(np.mean(time_angle))
        time_gt = []
        for restart in range(5):
            begin = time.time()
            K_laplace = cosine_distances(X)
            K_laplace = 1 - 2 / np.pi * (np.arccos(1 - K_laplace))
            time_gt.append(time.time() - begin)
        total_time_angle_gt.append(np.mean(time_gt))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    for ax, data, title in zip(
        axes,
        [
            [total_time_gauss, total_time_gauss_gt],
            [total_time_angle, total_time_angle_gt],
        ],
        ["Gaussian", "Angle"],
    ):
        ax.plot(np.arange(0, 5000, 200), data[0], label="approx")
        ax.plot(np.arange(0, 5000, 200), data[1], label="groundtruth")
        ax.set_xlabel("Number of samples")
        ax.set_ylabel("Time, S")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend()
    fig.suptitle("Boston Housing")
    plt.savefig("gauss_usps_time.png")
    plt.show()
