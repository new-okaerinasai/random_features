from random_features import RandomFeatures
from sklearn.metrics.pairwise import rbf_kernel, cosine_distances
import numpy as np
import matplotlib.pyplot as plt
N_SAMPLES, DIM = 1000, 200


if __name__ == "__main__":
    # size of data
    X = np.random.randn(N_SAMPLES, DIM)

    gamma = 2
    # Number of monte carlo samples D
    Ds = np.arange(1, 5000, 200)
    K_rbf, K_laplace = rbf_kernel(X, gamma=gamma), cosine_distances(X)
    K_laplace = 1 - 2 / np.pi * (np.arccos(1 - K_laplace))
    errors_rbf, errors_laplace = [], []

    for n in Ds:
        gauss = RandomFeatures(gamma=gamma, n=n, metric="rbf")
        gauss.fit(X)
        K_rbf_a = gauss.compute_kernel(X)

        angle = RandomFeatures(gamma=gamma, n=n, metric="angle")
        angle.fit(X)
        K_laplace_a = angle.compute_kernel(X)

        errors_rbf.append(((K_rbf_a - K_rbf) ** 2).mean())
        errors_laplace.append(((K_laplace_a - K_laplace) ** 2).mean())

    errors_rbf, errors_laplace = np.array(errors_rbf), np.array(errors_laplace)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    for ax, data, title in zip(
        axes, [errors_laplace, errors_rbf][::-1], ["Gaussian Kernel", "Angle Kernel"]
    ):
        ax.plot(Ds, data)
        ax.set_ylabel("MSE")
        ax.set_xlabel("Number of MC samples D")
        ax.set_yscale("log")
        ax.set_title(title)
    plt.savefig("fig.png")
    plt.show()
