import numpy as np

def least_squares(X, y, lr=0.001, iterations=10000, tol=1e-8):
    x = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    x_intercept = np.column_stack((np.ones(x.shape[0]), x))  # Add column of ones at index 0
    condition_number = np.linalg.cond(x_intercept)

    # Normal equations: beta = (X^T X)^{-1} X^T y  (exact closed-form solution)
    XtX = np.dot(np.transpose(x_intercept), x_intercept)
    Xty = np.dot(np.transpose(x_intercept), y)
    beta = np.linalg.solve(XtX, Xty)

    predictions = np.dot(x_intercept, beta)
    residuals = predictions - y
    final_loss = np.mean(residuals ** 2)

    payload = {
        "intercept": beta[0],
        "coefficients": beta[1:],
        "predictions": predictions,
        "mse": final_loss,
        "iterations": 1,
        "condition_number": condition_number,
    }
    return payload


def ridge_regression(X, y, lr=0.001, max_iter=10000, tol=1e-8, lamda=1.0):
    x = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    x_intercept = np.column_stack((np.ones(x.shape[0]), x))
    p = x_intercept.shape[1]

    penalty_diag = np.ones(p)
    penalty_diag[0] = 0.0
    D = np.diag(penalty_diag)

    XtX = np.dot(np.transpose(x_intercept), x_intercept)
    Xty = np.dot(np.transpose(x_intercept), y)
    beta = np.linalg.solve(XtX + lamda * D, Xty)

    predictions = np.dot(x_intercept, beta)
    residuals = predictions - y
    mse = np.mean(residuals ** 2)
    ridge_loss = mse + lamda * np.sum((penalty_diag * beta) ** 2)
    payload = {
        "intercept": beta[0],
        "coefficients": beta[1:],
        "predictions": predictions,
        "mse": mse,
        "ridge_loss": ridge_loss,
        "lambda": lamda,
        "iterations": 1,
    }
    return payload
