import numpy as np
import pytest
import scipy.linalg

from decompositions.lu import lu_decomposition
from decompositions.cholesky import cholesky
from decompositions.qr import qr_decomposition


def _spd_matrix(n=5, seed=42):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    return a @ a.T + np.eye(n)


def test_lu_reconstruct():
    A = np.array([[0., 2., 1.], [1., 1., 0.], [2., 0., 1.]])
    result = lu_decomposition(A)
    P, L, U = result["P"], result["L"], result["U"]
    assert np.allclose(P.T @ L @ U, A, atol=1e-5, rtol=1e-5)


def test_lu_L_is_lower_triangular():
    A = np.array([[0., 2., 1.], [1., 1., 0.], [2., 0., 1.]])
    L = lu_decomposition(A)["L"]
    assert np.allclose(L, np.tril(L), atol=1e-5, rtol=1e-5)
    assert np.allclose(np.diag(L), 1.0, atol=1e-5, rtol=1e-5)


def test_lu_U_is_upper_triangular():
    A = np.array([[0., 2., 1.], [1., 1., 0.], [2., 0., 1.]])
    U = lu_decomposition(A)["U"]
    assert np.allclose(U, np.triu(U), atol=1e-5, rtol=1e-5)


def test_lu_vs_scipy():
    A = np.array([[0., 2., 1.], [1., 1., 0.], [2., 0., 1.]])
    result = lu_decomposition(A)
    P, L, U = result["P"], result["L"], result["U"]
    P_ref, L_ref, U_ref = scipy.linalg.lu(A)
    # This project uses P.T @ L @ U = A; SciPy uses P_ref @ L_ref @ U_ref = A.
    assert np.allclose(P.T, P_ref, atol=1e-5, rtol=1e-5)
    assert np.allclose(L, L_ref, atol=1e-5, rtol=1e-5)
    assert np.allclose(U, U_ref, atol=1e-5, rtol=1e-5)


def test_lu_singular_raises():
    A = np.array([[1., 2.], [2., 4.]])
    with pytest.raises(ValueError):
        lu_decomposition(A)


def test_lu_non_square_raises():
    with pytest.raises(ValueError):
        lu_decomposition(np.ones((2, 3)))


def test_lu_3x3():
    A = np.array([[0., 2., 1.], [1., 1., 0.], [2., 0., 1.]])
    result = lu_decomposition(A)
    assert np.allclose(result["P"].T @ result["L"] @ result["U"], A,
                       atol=1e-5, rtol=1e-5)


def test_cholesky_reconstruct():
    A = _spd_matrix()
    L = cholesky(A)
    assert np.allclose(L @ L.T, A, atol=1e-5, rtol=1e-5)


def test_cholesky_L_lower_triangular():
    L = cholesky(_spd_matrix())
    assert np.allclose(L, np.tril(L), atol=1e-5, rtol=1e-5)


def test_cholesky_vs_scipy():
    A = _spd_matrix()
    L = cholesky(A)
    L_ref = scipy.linalg.cholesky(A, lower=True)
    assert np.allclose(L, L_ref, atol=1e-5, rtol=1e-5)


def test_cholesky_non_symmetric_raises():
    A = np.array([[2., 1.], [0., 2.]])
    with pytest.raises(ValueError):
        cholesky(A)


def test_cholesky_non_square_raises():
    with pytest.raises(ValueError):
        cholesky(np.ones((2, 3)))


def test_cholesky_identity():
    I = np.eye(4)
    assert np.allclose(cholesky(I), I, atol=1e-5, rtol=1e-5)


def test_qr_reconstruct():
    rng = np.random.default_rng(10)
    A = rng.normal(size=(8, 4))
    Q, R = qr_decomposition(A)
    assert np.allclose(Q @ R, A, atol=1e-5, rtol=1e-5)


def test_qr_Q_orthogonal():
    rng = np.random.default_rng(11)
    A = rng.normal(size=(8, 4))
    Q, _ = qr_decomposition(A)
    assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-5, rtol=1e-5)


def test_qr_R_upper_triangular():
    rng = np.random.default_rng(12)
    A = rng.normal(size=(8, 4))
    _, R = qr_decomposition(A)
    assert np.allclose(R, np.triu(R), atol=1e-5, rtol=1e-5)


def test_qr_vs_numpy():
    rng = np.random.default_rng(13)
    A = rng.normal(size=(8, 4))
    Q, R = qr_decomposition(A)
    Q_ref, R_ref = np.linalg.qr(A, mode="reduced")
    # QR factors are sign-ambiguous; align each column using R's diagonal.
    signs = np.sign(np.diag(R) * np.diag(R_ref))
    signs[signs == 0] = 1
    Q_ref = Q_ref * signs
    R_ref = signs[:, None] * R_ref
    assert np.allclose(Q, Q_ref, atol=1e-5, rtol=1e-5)
    assert np.allclose(R, R_ref, atol=1e-5, rtol=1e-5)


def test_qr_non_2d_raises():
    with pytest.raises(ValueError):
        qr_decomposition(np.array([1., 2., 3.]))


def test_qr_linearly_dependent_raises():
    A = np.array([[1., 2.], [2., 4.], [3., 6.]])
    with pytest.raises(ValueError):
        qr_decomposition(A)
