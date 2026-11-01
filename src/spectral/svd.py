import numpy as np


def truncated_svd(A, k, max_iter=10000, tol=1e-8):
    a = np.asarray(A, dtype=float)

    if k <= 0:
        raise ValueError("Value of k should be greater than 0")
    if a.ndim != 2:
        raise ValueError("Matrix should be of 2 Dimension")

    max_k = min(a.shape)
    if k > max_k:
        raise ValueError("k cannot exceed the number of available singular values.")

    m, n = a.shape
    B = np.dot(np.transpose(a), a)

    V = np.zeros((n, k))
    singular_values = np.zeros(k)

    for i in range(k):
        v = np.ones(n, dtype=float)
        v /= np.linalg.norm(v)

        for _ in range(max_iter):
            v_new = np.dot(B, v)

            norm_v = np.linalg.norm(v_new)
            if norm_v < tol:
                break

            v_new /= norm_v

            if min(
                np.linalg.norm(v_new - v),
                np.linalg.norm(v_new + v)
            ) < tol:
                v = v_new
                break

            v = v_new

        eigenvalue = float(np.dot(v, np.dot(B, v)))
        eigenvalue = max(eigenvalue, 0.0)
        sigma = np.sqrt(eigenvalue)

        singular_values[i] = sigma
        V[:, i] = v

        B = B - eigenvalue * np.outer(v, v)
        B = (B + np.transpose(B)) / 2

    order = np.argsort(singular_values)[::-1]
    singular_values = singular_values[order]
    V = V[:, order]

    U = np.zeros((m, k))
    for i in range(k):
        if singular_values[i] > tol:
            U[:, i] = np.dot(a, V[:, i]) / singular_values[i]

    Sigma = np.diag(singular_values)
    A_approx = np.dot(np.dot(U, Sigma), np.transpose(V))

    reconstruction_error = np.linalg.norm(a - A_approx, ord="fro")

    return {
        "U": U,
        "singular_values": singular_values,
        "Sigma": Sigma,
        "Vt": np.transpose(V),
        "A_approx": A_approx,
        "reconstruction_error": reconstruction_error,
    }
