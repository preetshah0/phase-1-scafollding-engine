import numpy as np
import pytest

from solvers.linear_solve import forward_substitution, backward_substitution
from solvers.conjugate_grad import conjugate_gradient
from solvers.least_squares import least_squares, ridge_regression
from decompositions.lu import lu_decomposition


ATOL = 1e-5
RTOL = 1e-5


def test_forward_substitution_known():
    L = np.array([[2., 0., 0.], [1., 3., 0.], [2., -1., 1.]])
    b = np.array([2., 7., 3.])
    x = forward_substitution(L, b)
    assert np.allclose(L @ x, b, atol=ATOL, rtol=RTOL)


def test_backward_substitution_known():
    U = np.array([[2., 1., 1.], [0., 3., -1.], [0., 0., 2.]])
    b = np.array([4., 5., 2.])
    x = backward_substitution(U, b)
    assert np.allclose(U @ x, b, atol=ATOL, rtol=RTOL)


def test_forward_vs_numpy():
    L = np.array([[2., 0., 0.], [1., 3., 0.], [2., -1., 1.]])
    b = np.array([2., 7., 3.])
    assert np.allclose(forward_substitution(L, b), np.linalg.solve(L, b),
                       atol=ATOL, rtol=RTOL)


def test_backward_vs_numpy():
    U = np.array([[2., 1., 1.], [0., 3., -1.], [0., 0., 2.]])
    b = np.array([4., 5., 2.])
    assert np.allclose(backward_substitution(U, b), np.linalg.solve(U, b),
                       atol=ATOL, rtol=RTOL)


def test_lu_full_solve():
    A = np.array([[0., 2., 1.], [1., 1., 0.], [2., 0., 1.]])
    b = np.array([1., 2., 3.])
    factors = lu_decomposition(A)
    P, L, U = factors["P"], factors["L"], factors["U"]
    # P @ A = L @ U for the project's P.T @ L @ U = A convention.
    Pb = P @ b
    z = forward_substitution(L, Pb)
    x = backward_substitution(U, z)
    assert np.allclose(A @ x, b, atol=ATOL, rtol=RTOL)


def _spd(n=10, seed=7):
    rng = np.random.default_rng(seed)
    M = rng.normal(size=(n, n))
    return M.T @ M + np.eye(n)


def _cg(A, b, x0=None):
    # Adapt this helper if your solver uses a different x0 keyword.
    if x0 is None:
        return conjugate_gradient(A, b)
    return conjugate_gradient(A, b, x0=x0)


def test_cg_solves_spd():
    A = _spd()
    b = np.arange(1., 11.)
    x = _cg(A, b)
    assert np.allclose(A @ x, b, atol=ATOL, rtol=RTOL)


def test_cg_residual_small():
    A = _spd(seed=8)
    b = np.arange(1., 11.)
    x = _cg(A, b)
    assert np.linalg.norm(A @ x - b) < 1e-6


def test_cg_vs_numpy():
    A = _spd(seed=9)
    b = np.arange(1., 11.)
    x = _cg(A, b)
    assert np.allclose(x, np.linalg.solve(A, b), atol=ATOL, rtol=RTOL)


def test_cg_with_x0():
    A = _spd(seed=10)
    b = np.arange(1., 11.)
    x0 = np.ones(10)
    x = _cg(A, b, x0=x0)
    assert np.allclose(A @ x, b, atol=ATOL, rtol=RTOL)


def test_cg_identity():
    A = np.eye(5)
    b = np.array([1., 2., 3., 4., 5.])
    x = _cg(A, b)
    assert np.allclose(x, b, atol=ATOL, rtol=RTOL)


def _linear_data():
    X = np.arange(1., 11.).reshape(-1, 1)
    y = 3.0 + 2.5 * X[:, 0]
    return X, y


def test_least_squares_perfect_fit():
    X, y = _linear_data()
    result = least_squares(X, y)
    assert result["mse"] < 1e-5


def test_least_squares_coefficients():
    X, y = _linear_data()
    result = least_squares(X, y)
    assert np.allclose(np.asarray(result["coefficients"]), [2.5],
                       atol=0.05, rtol=0.05)


def test_least_squares_predictions_shape():
    X, y = _linear_data()
    result = least_squares(X, y)
    assert np.asarray(result["predictions"]).shape == y.shape


def test_least_squares_mse_positive():
    rng = np.random.default_rng(21)
    X = rng.normal(size=(30, 2))
    y = 1.5 + X @ np.array([2., -0.5]) + rng.normal(scale=0.1, size=30)
    result = least_squares(X, y)
    assert result["mse"] >= 0


def test_ridge_regression_reduces_overfitting():
    # A high-dimensional, noisy design makes unregularized fitting more variable.
    rng = np.random.default_rng(22)
    X_train = rng.normal(size=(18, 12))
    true_beta = np.zeros(12)
    true_beta[:3] = [2.0, -1.5, 0.8]
    y_train = 1.0 + X_train @ true_beta + rng.normal(scale=1.0, size=18)

    X_test = rng.normal(size=(200, 12))
    y_test = 1.0 + X_test @ true_beta

    ols = least_squares(X_train, y_train)
    ridge = ridge_regression(X_train, y_train, lr=0.001,
                             max_iter=30000, tol=1e-10, lamda=1.0)

    ols_pred = ols["intercept"] + X_test @ np.asarray(ols["coefficients"])
    ridge_pred = ridge["intercept"] + X_test @ np.asarray(ridge["coefficients"])
    ols_test_mse = np.mean((ols_pred - y_test) ** 2)
    ridge_test_mse = np.mean((ridge_pred - y_test) ** 2)
    assert ridge_test_mse < ols_test_mse


def test_ridge_coefficients_shrink():
    rng = np.random.default_rng(23)
    X = rng.normal(size=(60, 5))
    y = 2.0 + X @ np.array([3., -2., 1., 0.5, -1.]) + rng.normal(scale=0.5, size=60)

    ols = least_squares(X, y)
    ridge = ridge_regression(X, y, lr=0.001, max_iter=30000,
                             tol=1e-10, lamda=10.0)

    ols_norm = np.linalg.norm(np.asarray(ols["coefficients"]))
    ridge_norm = np.linalg.norm(np.asarray(ridge["coefficients"]))
    assert ridge_norm < ols_norm
