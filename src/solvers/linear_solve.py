import numpy as np


def forward_substitution(L, b):
    """Solves L * x = b where L is a lower triangular matrix."""
    n = L.shape[0]
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(n):
        s = np.dot(L[i, :i], x[:i])
        x[i] = (b[i] - s) / L[i, i]
    return x


def backward_substitution(U, b):
    """Solves U * x = b where U is an upper triangular matrix."""
    n = U.shape[0]
    x = np.zeros_like(b, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = np.dot(U[i, i + 1 :], x[i + 1 :])
        x[i] = (b[i] - s) / U[i, i]
    return x
