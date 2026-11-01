import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from decompositions.cholesky import cholesky
from solvers.linear_solve import forward_substitution, backward_substitution

def noise_generator(n_samples):
    np.random.seed(42)

    x = np.linspace(0, 2 * np.pi, n_samples)

    y_true = np.sin(x)
    noise = np.random.normal(loc=0.0, scale=0.2, size=n_samples)
    y = y_true + noise
    return x, y

def rbf_kernel(X1, X2, sigma=1.0, length_scale=1.0):
    X1 = np.asarray(X1).reshape(-1, 1)
    X2 = np.asarray(X2).reshape(-1, 1)

    squared_distances = (X1 - np.transpose(X2)) ** 2

    K = (sigma ** 2) * np.exp(
        -squared_distances / (2 * length_scale ** 2)
    )

    return K

def gp_regression(n_samples, sigma, length_scale, sigma_n):
    x, y = noise_generator(n_samples)

    K = rbf_kernel(x, x, sigma=1.0, length_scale=1.0)

    K_noise = K + (sigma**2) * np.eye(n_samples)

    print("GP Regression on 20 noisy sine observations.")
    print(f"Kernel: RBF | Length-scale: {length_scale} | "
          f"Signal variance: {sigma**2} | Noise: {sigma_n}")
    print("Solving via Cholesky factorization...")

    L = cholesky(K_noise)

    z = forward_substitution(L, y)

    alpha = backward_substitution(np.transpose(L), z)

    log_likelihood = (
        -0.5 * (np.dot(y, alpha))
        - np.sum(np.log(np.diag(L)))
        - 0.5 * n_samples * np.log(2 * np.pi)
    )

    print(f"\nLog marginal likelihood: {log_likelihood:.2f}")

    x_test = np.linspace(0, 2 * np.pi, 100)

    K_star = rbf_kernel(
        x, x_test,
        sigma=sigma,
        length_scale=length_scale
    )

    mean = np.dot(np.transpose(K_star), alpha)

    # Solve L V = K_star
    V = np.zeros_like(K_star)

    for j in range(K_star.shape[1]):
        V[:, j] = forward_substitution(L, K_star[:, j])

    # Predictive variance
    K_test = rbf_kernel(
        x_test, x_test,
        sigma=sigma,
        length_scale=length_scale
    )

    variance = np.diag(K_test) - np.sum(V**2, axis=0)

    variance = np.maximum(variance, 0.0)

    std = np.sqrt(variance)

    print("\nPredictions at 100 test points computed.")

    # Plot
    plt.figure(figsize=(10, 6))

    plt.scatter(
        x, y,
        color="blue",
        label="Training observations"
    )

    plt.plot(
        x_test, mean,
        color="red",
        label="GP predicted mean"
    )

    plt.fill_between(
        x_test,
        mean - 2 * std,
        mean + 2 * std,
        alpha=0.2,
        label="±2σ uncertainty"
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gaussian Process Regression")
    plt.legend()
    plt.tight_layout()

    # Save plot
    output_path = Path("outputs")
    output_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(
        output_path / "gp_regression.png",
        dpi=150
    )

    plt.close()

    print("Plot saved to: outputs/gp_regression.png")

    return x, y, x_test, mean, variance
