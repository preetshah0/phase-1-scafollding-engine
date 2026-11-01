import numpy as np

def qr_decomposition(A):
    a = np.asarray(A, dtype=float)

    if a.ndim != 2:
        raise ValueError("Matrix must be 2D.")

    m, n = a.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    # Classical Gram-Schmidt Process
    for j in range(n):
        v = a[:, j].copy()

        for i in range(j):
            R[i, j] = np.dot(Q[:, i], a[:, j])
            v -= R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)

        if np.isclose(R[j, j], 0.0):
            raise ValueError("Matrix columns are linearly dependent; QR decomposition failed.")

        Q[:, j] = v / R[j, j]

    return Q, R
