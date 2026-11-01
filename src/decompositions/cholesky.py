import numpy as np

def cholesky(A):
    a = np.asarray(A, dtype=float)
    if a.ndim != 2:
        raise ValueError("Matrix must be 2D.")

    if a.shape[0] != a.shape[1]:
        raise ValueError("Matrix must be square.")

    if a.shape[0] == 0:
        raise ValueError("Matrix cannot be empty.")

    if not np.all(np.isfinite(a)):
        raise ValueError("Matrix must contain finite values.")

    if not np.allclose(a, a.T):
        raise ValueError("Matrix must be symmetric.")
    n = a.shape[0]
    L = np.zeros((n,n))
    for i in range(n):
        for j in range(i + 1):  
        
            if i == j:  
                L[i][i] = np.sqrt(a[i][i] - sum(L[i][k]**2 for k in range(i)))
            
            else:  
                L[i][j] = (a[i][j] - sum(L[i][k] * L[j][k] for k in range(j))) / L[j][j]
    return L