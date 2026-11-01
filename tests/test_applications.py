import numpy as np
import pytest
from pathlib import Path

from spectral.svd import truncated_svd
from applications.gp_regression import gp_regression


ATOL = 1e-5
RTOL = 1e-5


def _svd_result(A, k):
    return truncated_svd(A, k)


def test_svd_shapes():
    rng = np.random.default_rng(31)
    A = rng.normal(size=(12, 8))
    result = _svd_result(A, 4)
    assert result["U"].shape == (12, 4)
    assert result["Sigma"].shape == (4, 4)
    assert result["Vt"].shape == (4, 8)
    assert result["A_approx"].shape == (12, 8)


def test_svd_reconstruction_matches_numpy():
    rng = np.random.default_rng(32)
    A = rng.normal(size=(10, 7))
    result = _svd_result(A, 3)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    A_ref = U[:, :3] @ np.diag(s[:3]) @ Vt[:3, :]
    err = np.linalg.norm(A - result["A_approx"], ord="fro")
    ref_err = np.linalg.norm(A - A_ref, ord="fro")
    assert np.isclose(err, ref_err, atol=ATOL, rtol=RTOL)


def test_svd_singular_values_match_numpy():
    rng = np.random.default_rng(33)
    A = rng.normal(size=(10, 7))
    result = _svd_result(A, 3)
    s_ref = np.linalg.svd(A, compute_uv=False)[:3]
    assert np.allclose(result["singular_values"], s_ref,
                       atol=ATOL, rtol=RTOL)


def test_svd_error_decreases_with_rank():
    rng = np.random.default_rng(34)
    A = rng.normal(size=(20, 15))
    e2 = _svd_result(A, 2)["reconstruction_error"]
    e5 = _svd_result(A, 5)["reconstruction_error"]
    e10 = _svd_result(A, 10)["reconstruction_error"]
    assert e10 < e5 < e2


def test_svd_k_exceeds_rank_raises():
    with pytest.raises(ValueError):
        _svd_result(np.ones((5, 4)), 5)


def test_svd_k_zero_raises():
    with pytest.raises(ValueError):
        _svd_result(np.ones((5, 4)), 0)


def test_svd_image_compression_shape():
    rng = np.random.default_rng(35)
    channel = rng.integers(0, 256, size=(100, 80)).astype(float)
    result = _svd_result(channel, 10)
    assert result["A_approx"].shape == (100, 80)


def _call_gp(n=20):
    # The function signature in the project specification is positional.
    return gp_regression(n, 1.0, 1.0, 0.2)


def test_gp_output_shapes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    x, y, x_test, mean, variance = _call_gp(20)
    assert x.shape == (20,)
    assert y.shape == (20,)
    assert x_test.shape == (100,)
    assert mean.shape == (100,)
    assert variance.shape == (100,)


def test_gp_variance_nonnegative(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, _, _, _, variance = _call_gp(20)
    assert np.all(variance >= 0)


def test_gp_mean_in_plausible_range(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _, _, _, mean, _ = _call_gp(20)
    assert np.all(mean >= -2.0)
    assert np.all(mean <= 2.0)


def test_gp_plot_saved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _call_gp(20)
    assert (Path("outputs") / "gp_regression.png").exists()
