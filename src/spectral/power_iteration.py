import numpy as np

def power_iteration(A,lamda,x0,tol=1e-8, max_iter=10000):
    a = np.asarray(A, dtype=float)
    if a.ndim !=2:
        raise ValueError("Matrix should be of 2D")
    if a.shape[0] != a.shape[1]:
        raise ValueError("Matrix must be square.")

    if a.shape[0] == 0:
        raise ValueError("Matrix cannot be empty.")

    if not np.all(np.isfinite(a)):
        raise ValueError("Matrix must contain finite values.")

    if max_iter < 1 or tol <= 0:
        raise ValueError("Invalid iteration parameters.")

    n = a.shape[0]
    if x0 is None:
        x = np.zeros((n,n))
    else:
        x = np.asarray(x0, dtype=float)

    if x.ndim != 1 or len(x) != n:
            raise ValueError("Initial vector must have shape (n,).")

    if not np.all(np.isfinite(x)):
            raise ValueError("Initial vector must contain finite values.")

    norm_x = np.linalg.norm(x)

    if norm_x == 0:
        raise ValueError("Initial vector cannot be zero.")

    x = x / norm_x

    
    converged = False
    eigenvalue = 0.0

    for itr in range(max_iter):

        x_new = np.dot(a,x)

        norm_x = np.linalg.norm(x_new)

        if norm_x == 0:
            raise ValueError("Cannot continue: matrix produced a zero vector.")

        
        x_new = x_new / norm_x

        error = min(
            np.linalg.norm(x_new - x),
            np.linalg.norm(x_new + x)
        )

        x = x_new

        if error < tol:
            converged = True
            break

    #Calculate Rayleigh quotient
    eigenvalue = np.dot(np.transpose(x), a , x) / np.dot(np.transpose(x),x)


    payload = {
        "eigenvalue": eigenvalue,
        "eigenvector": x,
        "iterations": itr + 1,
        "converged": converged
    }

    return payload
    
