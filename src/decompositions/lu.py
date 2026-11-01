import numpy as np

def lu_decomposition(A):
    a = np.asarray(A, dtype=float)
    if a.ndim != 2:
        raise ValueError("Matrix must be 2D.")

    if a.shape[0] != a.shape[1]:
        raise ValueError("Matrix must be square.")

    if not np.all(np.isfinite(a)):
        raise ValueError("Matrix must contain finite values.")

    n = a.shape[0]

    u = np.copy(a)
    l = np.eye(n)
    p = np.eye(n)

    for i in range(n - 1):

        pivot_row = i + np.argmax(np.abs(u[i:, i]))

        if np.isclose(u[pivot_row, i], 0.0):
            raise ValueError("Matrix is singular or numerically singular.")

        if pivot_row != i:
            u[[i, pivot_row], :] = u[[pivot_row, i], :]
            p[[i, pivot_row], :] = p[[pivot_row, i], :]

            l[[i, pivot_row], :i] = l[[pivot_row, i], :i]

        for j in range(i + 1, n):
            factor = u[j, i] / u[i, i]
            l[j, i] = factor
            u[j, :] = u[j, :] - factor * u[i, :]

    
    if np.isclose(u[n - 1, n - 1], 0.0):
        raise ValueError("Matrix is singular or numerically singular.")

    payload =  {
        "P": p,
        "L": l,
        "U": u
    }
    return payload