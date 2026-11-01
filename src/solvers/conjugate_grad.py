import numpy as np


def conjugate_gradient(A, b, x0=None, tol=1e-8, max_iter=1000):
  """Solves Ax = b using the Conjugate Gradient method."""
  n = len(b)
  if x0 is None:
    x = np.zeros(n)
  else:
    x = np.copy(x0)

  r = b - np.dot(A, x)
  p = np.copy(r)
  r_dot_old = np.dot(r, r)

  for i in range(max_iter):
    # If residual is close enough to zero, we found the solution
    if np.sqrt(r_dot_old) < tol:
      print(f"Converged in {i} iterations.")
      return x

    Ap = np.dot(A, p)
    alpha = r_dot_old / np.dot(p, Ap)

    x += alpha * p
    r -= alpha * Ap

    r_dot_new = np.dot(r, r)
    beta = r_dot_new / r_dot_old
    p = r + beta * p

    r_dot_old = r_dot_new

  print("Reached max iterations without full convergence.")
  return x
