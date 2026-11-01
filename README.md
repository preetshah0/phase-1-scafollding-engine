# Computational Linear Algebra & Math Engine for ML

> **Phase 1 — AI Engineering from Scratch**  
> A modular, zero-dependency computational mathematics engine implementing foundational algorithms of AI/ML from scratch using Python and NumPy.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0%2B-orange)](https://numpy.org/)
[![Tests](https://img.shields.io/badge/Tests-46%20passed-brightgreen)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

---

## Table of Contents

1. [Repository Access & Setup](#1-repository-access--setup)
2. [Project Structure](#2-project-structure)
3. [Module Reference](#3-module-reference)
   - [LU Decomposition](#31-lu-decomposition)
   - [Cholesky Decomposition](#32-cholesky-decomposition)
   - [QR Decomposition](#33-qr-decomposition)
   - [Forward & Backward Substitution](#34-forward--backward-substitution)
   - [Conjugate Gradient Solver](#35-conjugate-gradient-solver)
   - [Least Squares Regression](#36-least-squares-regression)
   - [Ridge Regression](#37-ridge-regression)
   - [Truncated SVD](#38-truncated-svd)
   - [SVD Image Compression](#39-svd-image-compression)
   - [Gaussian Process Regression](#310-gaussian-process-regression)
4. [Running the Demo Notebook](#4-running-the-demo-notebook)
5. [Running Tests](#5-running-tests)

---

## 1. Repository Access & Setup

### Clone the repository

```bash
git clone https://github.com/preetshah0/phase-1-scafollding-engine.git
cd phase-1-scafollding-engine
```

### Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -e ".[dev]"
# or manually:
pip install numpy>=2.0.0 matplotlib>=3.8.0 pytest>=9.2.0 opencv-python jupyter
```

### Add `src/` to the Python path

The source modules live under `src/`. Set `PYTHONPATH` before running any script or test:

```bash
# Windows PowerShell
$env:PYTHONPATH = "src"

# macOS / Linux
export PYTHONPATH=src
```

> **Tip — VS Code users:** add `"python.envFile": "${workspaceFolder}/.env"` and create a `.env` file containing `PYTHONPATH=src`.

---

## 2. Project Structure

```
phase-1-dev-scaffolding-engine/
├── src/
│   ├── decompositions/
│   │   ├── lu.py               # LU decomposition with partial pivoting
│   │   ├── cholesky.py         # Cholesky decomposition (SPD matrices)
│   │   └── qr.py               # QR decomposition (Gram-Schmidt)
│   ├── solvers/
│   │   ├── linear_solve.py     # Forward & backward substitution
│   │   ├── conjugate_grad.py   # Conjugate Gradient method
│   │   └── least_squares.py    # OLS & Ridge regression (normal equations)
│   ├── spectral/
│   │   ├── power_iteration.py  # Power iteration & Rayleigh quotient
│   │   └── svd.py              # Truncated SVD via power iteration
│   └── applications/
│       ├── svd_compression.py  # Image compression using low-rank SVD
│       └── gp_regression.py    # Gaussian Process Regression (RBF kernel)
├── tests/
│   ├── test_decompositions.py
│   ├── test_solvers.py
│   └── test_applications.py
├── notebooks/
│   └── demo_engine.ipynb       # Interactive end-to-end walkthrough
├── photo.jfif                  # Sample image for SVD demo
└── pyproject.toml
```

---

## 3. Module Reference

---

### 3.1 LU Decomposition

**File:** [`src/decompositions/lu.py`](src/decompositions/lu.py)

#### Concept

LU decomposition factors a square matrix **A** into a permutation matrix **P**, a lower-triangular matrix **L**, and an upper-triangular matrix **U** such that:

$$PA = LU$$

Partial pivoting (swapping rows to put the largest-magnitude entry on the diagonal) is used at each step for numerical stability.

#### Formula — Elimination step

At column $i$, for each row $j > i$:

$$m_{ji} = \frac{u_{ji}}{u_{ii}}, \qquad \text{row}_j \leftarrow \text{row}_j - m_{ji} \cdot \text{row}_i$$

The multipliers $m_{ji}$ populate **L**; the modified rows form **U**.

#### Application

- Solve $Ax = b$ in $O(n^2)$ after an $O(n^3)$ factorisation (reuse **L**, **U** for many right-hand sides).
- Compute $\det(A) = \det(U)$ (product of diagonal entries, sign from pivot count).
- Foundation for matrix inversion.

#### Implementation Steps

1. Copy **A** into **U**; initialise **L** = **I**, **P** = **I**.
2. For column $i = 0 \ldots n-2$:
   - Find pivot row: $p = \arg\max_{k \geq i} |U_{ki}|$.
   - Swap rows $i$ and $p$ in **U**, **P**, and the already-filled part of **L**.
   - For each row $j > i$: compute multiplier $m = U_{ji} / U_{ii}$, store in $L_{ji}$, zero out $U_{ji}$.
3. Return `{"P": P, "L": L, "U": U}`.

#### Usage

```python
from decompositions.lu import lu_decomposition

result = lu_decomposition(A)
P, L, U = result["P"], result["L"], result["U"]
print(f"‖PA − LU‖ = {np.linalg.norm(P @ A - L @ U):.2e}")
```

---

### 3.2 Cholesky Decomposition

**File:** [`src/decompositions/cholesky.py`](src/decompositions/cholesky.py)

#### Concept

For a **symmetric positive-definite (SPD)** matrix **A**, Cholesky factorisation finds a lower-triangular **L** such that:

$$A = LL^\top$$

It is twice as fast as LU decomposition and numerically more stable for SPD matrices.

#### Formulae

Diagonal entries:

$$L_{ii} = \sqrt{A_{ii} - \sum_{k=0}^{i-1} L_{ik}^2}$$

Off-diagonal entries ($j < i$):

$$L_{ij} = \frac{1}{L_{jj}}\left(A_{ij} - \sum_{k=0}^{j-1} L_{ik} L_{jk}\right)$$

#### Application

- Solving SPD linear systems (e.g. normal equations in regression, kernel systems in GP).
- Sampling from multivariate Gaussians: $x = L z,\; z \sim \mathcal{N}(0, I)$.
- Computing log-determinants for likelihood calculations.

#### Implementation Steps

1. Validate: square, symmetric, finite, non-empty.
2. Allocate **L** = zeros($n \times n$).
3. For $i = 0 \ldots n-1$, for $j = 0 \ldots i$:
   - If $i = j$: $L_{ii} = \sqrt{A_{ii} - \sum_k L_{ik}^2}$.
   - Else: $L_{ij} = (A_{ij} - \sum_k L_{ik} L_{jk}) / L_{jj}$.
4. Return **L**.

#### Usage

```python
from decompositions.cholesky import cholesky

A = M.T @ M + np.eye(n)   # make it SPD
L = cholesky(A)
print(f"‖LLᵀ − A‖ = {np.linalg.norm(L @ L.T - A):.2e}")
```

---

### 3.3 QR Decomposition

**File:** [`src/decompositions/qr.py`](src/decompositions/qr.py)

#### Concept

Factors a tall matrix $A \in \mathbb{R}^{m \times n}$ ($m \geq n$) into an orthonormal matrix $Q \in \mathbb{R}^{m \times n}$ and an upper-triangular matrix $R \in \mathbb{R}^{n \times n}$:

$$A = QR, \qquad Q^\top Q = I_n$$

Uses the **classical Gram-Schmidt** orthogonalisation process.

#### Formulae

For column $j$, starting from $v = a_j$:

$$R_{ij} = q_i^\top a_j \quad (i < j), \qquad v \leftarrow v - R_{ij}\, q_i$$

$$R_{jj} = \|v\|, \qquad q_j = v / R_{jj}$$

#### Application

- Solving least-squares problems: $\min_x \|Ax - b\|$.
- Eigenvalue algorithms (QR iteration).
- Orthonormal basis construction.

#### Implementation Steps

1. Validate: 2-D, $m \geq n$, columns linearly independent.
2. Allocate $Q$, $R$ as zeros.
3. For $j = 0 \ldots n-1$:
   - $v = a_j$ (copy of column $j$).
   - For $i < j$: $R_{ij} = q_i^\top a_j$; $v \leftarrow v - R_{ij} q_i$.
   - $R_{jj} = \|v\|$; if $\approx 0$ → raise (dependent columns).
   - $q_j = v / R_{jj}$.
4. Return `Q, R`.

#### Usage

```python
from decompositions.qr import qr_decomposition

Q, R = qr_decomposition(A)
print(f"‖QR − A‖ = {np.linalg.norm(Q @ R - A):.2e}")
print(f"‖QᵀQ − I‖ = {np.linalg.norm(Q.T @ Q - np.eye(Q.shape[1])):.2e}")
```

---

### 3.4 Forward & Backward Substitution

**File:** [`src/solvers/linear_solve.py`](src/solvers/linear_solve.py)

#### Concept

Used to solve triangular systems arising from decompositions.

- **Forward substitution** solves $Lx = b$ (lower-triangular **L**).
- **Backward substitution** solves $Ux = b$ (upper-triangular **U**).

#### Formulae

Forward ($i = 0, 1, \ldots, n-1$):

$$x_i = \frac{b_i - \sum_{k=0}^{i-1} L_{ik}\, x_k}{L_{ii}}$$

Backward ($i = n-1, n-2, \ldots, 0$):

$$x_i = \frac{b_i - \sum_{k=i+1}^{n-1} U_{ik}\, x_k}{U_{ii}}$$

#### Application

- Core building block for all decomposition-based solvers (LU, Cholesky, QR).
- Each solve is $O(n^2)$, vs $O(n^3)$ for full Gaussian elimination.

#### Implementation Steps

1. Allocate $x$ = zeros($n$).
2. Forward: iterate $i$ from 0 to $n-1$; accumulate dot product over already-solved entries.
3. Backward: iterate $i$ from $n-1$ to 0; accumulate dot product over already-solved entries.
4. Return $x$.

#### Usage

```python
from solvers.linear_solve import forward_substitution, backward_substitution

x = forward_substitution(L, b)     # Lx = b
x = backward_substitution(U, b)    # Ux = b
```

---

### 3.5 Conjugate Gradient Solver

**File:** [`src/solvers/conjugate_grad.py`](src/solvers/conjugate_grad.py)

#### Concept

An iterative method for solving $Ax = b$ where **A** is **symmetric positive-definite**. Converges in at most $n$ steps (exact arithmetic) and often far fewer in practice.

#### Formulae

Initialise $x_0$, $r_0 = b - Ax_0$, $p_0 = r_0$.

Each iteration $i$:

$$\alpha_i = \frac{r_i^\top r_i}{p_i^\top A p_i}, \qquad x_{i+1} = x_i + \alpha_i p_i$$

$$r_{i+1} = r_i - \alpha_i A p_i, \qquad \beta_i = \frac{r_{i+1}^\top r_{i+1}}{r_i^\top r_i}$$

$$p_{i+1} = r_{i+1} + \beta_i p_i$$

#### Application

- Large sparse SPD systems where LU factorisation is expensive.
- Normal equations $(X^\top X)w = X^\top y$ in high-dimensional regression.
- Solving kernel systems in GP regression.

#### Implementation Steps

1. Initialise $x = x_0$ (zeros if not provided), $r = b - Ax$, $p = r$.
2. Loop up to `max_iter`:
   - Check $\|r\| < \text{tol}$ → converged, return $x$.
   - Compute $Ap$, $\alpha = r^\top r / (p^\top Ap)$.
   - Update $x$, $r$, compute $\beta$, update $p$.
3. Return $x$.

#### Usage

```python
from solvers.conjugate_grad import conjugate_gradient

x = conjugate_gradient(A, b)               # zeros start
x = conjugate_gradient(A, b, x0=x0)       # warm start
print(f"Residual: {np.linalg.norm(A @ x - b):.2e}")
```

---

### 3.6 Least Squares Regression

**File:** [`src/solvers/least_squares.py`](src/solvers/least_squares.py)

#### Concept

Fits a linear model $\hat{y} = X\beta$ by minimising the mean squared error:

$$\min_\beta \|X\beta - y\|^2$$

Solved via the **normal equations** — an exact, closed-form solution.

#### Formula

$$\hat{\beta} = (X^\top X)^{-1} X^\top y$$

The design matrix $\tilde{X}$ prepends a column of ones to model the intercept.

#### Application

- Linear regression and curve fitting.
- Polynomial regression (with feature expansion).
- Baseline model for comparing against regularised variants.

#### Implementation Steps

1. Prepend a ones column to $X$ → $\tilde{X}$.
2. Compute $\tilde{X}^\top \tilde{X}$ and $\tilde{X}^\top y$.
3. Solve via `np.linalg.solve` (numerically stable vs explicit inverse).
4. Compute predictions $\hat{y} = \tilde{X}\hat{\beta}$ and MSE.
5. Return `{"intercept", "coefficients", "predictions", "mse", ...}`.

#### Usage

```python
from solvers.least_squares import least_squares

result = least_squares(X, y)
print(f"Intercept: {result['intercept']:.4f}")
print(f"Coefs:     {result['coefficients']}")
print(f"MSE:       {result['mse']:.4f}")
```

---

### 3.7 Ridge Regression

**File:** [`src/solvers/least_squares.py`](src/solvers/least_squares.py)

#### Concept

Adds an L2 penalty $\lambda\|\beta\|^2$ to OLS to shrink coefficients and reduce overfitting, particularly in high-dimensional or ill-conditioned settings.

#### Formula

$$\hat{\beta}_\text{ridge} = (X^\top X + \lambda D)^{-1} X^\top y$$

where $D = \text{diag}(0, 1, 1, \ldots, 1)$ — the intercept is **not** penalised.

#### Application

- Regularised regression when $p \approx n$ or $p > n$.
- Prevents coefficient blow-up in multicollinear designs.
- Controls the bias–variance trade-off via $\lambda$.

#### Implementation Steps

1. Prepend ones column → $\tilde{X}$.
2. Build penalty diagonal $D$ (zero for intercept index, one for all others).
3. Solve $(\tilde{X}^\top\tilde{X} + \lambda D)\hat{\beta} = \tilde{X}^\top y$ via `np.linalg.solve`.
4. Compute MSE and ridge loss.
5. Return `{"intercept", "coefficients", "mse", "ridge_loss", "lambda", ...}`.

#### Usage

```python
from solvers.least_squares import ridge_regression

result = ridge_regression(X, y, lamda=1.0)
print(f"Ridge MSE:  {result['mse']:.4f}")
print(f"Ridge loss: {result['ridge_loss']:.4f}")
```

---

### 3.8 Truncated SVD

**File:** [`src/spectral/svd.py`](src/spectral/svd.py)

#### Concept

Decomposes a matrix $A \in \mathbb{R}^{m \times n}$ into:

$$A \approx U_k \Sigma_k V_k^\top$$

where $U_k \in \mathbb{R}^{m \times k}$, $\Sigma_k \in \mathbb{R}^{k \times k}$ (diagonal), $V_k \in \mathbb{R}^{n \times k}$ are the top-$k$ singular triplets. The approximation is optimal in the Frobenius and spectral norms (Eckart–Young theorem).

#### Key Formulae

Singular values are square-roots of eigenvalues of $A^\top A$:

$$A^\top A\, v_i = \sigma_i^2\, v_i, \qquad \sigma_i = \sqrt{\lambda_i(A^\top A)}$$

Left singular vectors:

$$u_i = \frac{A v_i}{\sigma_i}$$

Deflation after extracting each eigenpair:

$$B \leftarrow B - \lambda_i\, v_i v_i^\top$$

Compression ratio (stored values vs original):

$$\text{compression} = \left(1 - \frac{k(m + n + 1)}{mn}\right) \times 100\%$$

#### Application

- **Image compression** — keep only the top-$k$ singular triplets.
- **Dimensionality reduction** (PCA).
- **Latent Semantic Analysis** in NLP.
- **Noise filtering** in signal processing.

#### Implementation Steps

1. Compute $B = A^\top A$.
2. For $i = 1 \ldots k$:
   - Power-iterate on $B$ to find eigenvector $v_i$ and eigenvalue $\lambda_i$.
   - $\sigma_i = \sqrt{\max(\lambda_i, 0)}$; store $v_i$, $\sigma_i$.
   - Deflate: $B \leftarrow B - \lambda_i v_i v_i^\top$; re-symmetrise.
3. Sort by descending $\sigma$.
4. Recover $U_k$: for each column $i$, $u_i = A v_i / \sigma_i$.
5. Reconstruct $\hat{A} = U_k \Sigma_k V_k^\top$; compute Frobenius reconstruction error.
6. Return `{"U", "singular_values", "Sigma", "Vt", "A_approx", "reconstruction_error"}`.

#### Usage

```python
from spectral.svd import truncated_svd

result = truncated_svd(A, k=20)
print(f"Reconstruction error: {result['reconstruction_error']:.4f}")
A_approx = result["A_approx"]
```

---

### 3.9 SVD Image Compression

**File:** [`src/applications/svd_compression.py`](src/applications/svd_compression.py)

#### Concept

Applies truncated SVD independently to each colour channel (R, G, B) of an image, then merges the reconstructed channels. Lower rank → higher compression, lower quality.

#### Quality Metric — PSNR

$$\text{PSNR} = 20 \log_{10}\!\left(\frac{255}{\sqrt{\text{MSE}}}\right) \text{ dB}$$

| PSNR | Quality |
|------|---------|
| ≥ 35 dB | Near-original |
| 25–35 dB | Decent |
| < 25 dB | Visibly blurry |

#### Implementation Steps

1. Load the image with OpenCV; split into B, G, R channels.
2. For each rank $k \in \{5, 20, 50\}$:
   - Apply `truncated_svd(channel, k)` to each channel.
   - Merge reconstructed channels; clip to [0, 255].
   - Compute compression % and PSNR; print summary.
   - Save to `outputs/compression/rank_{k}.png`.

---

### 3.10 Gaussian Process Regression

**File:** [`src/applications/gp_regression.py`](src/applications/gp_regression.py)

#### Concept

A non-parametric Bayesian regression model that places a prior over functions. Given training data $(X, y)$, the posterior predictive distribution at test points $X_*$ is Gaussian with computable mean and variance — giving both predictions and calibrated uncertainty.

#### RBF (Squared-Exponential) Kernel

$$k(x, x') = \sigma^2 \exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right)$$

- $\sigma^2$ — signal variance (output scale).
- $\ell$ — length-scale (controls smoothness; larger = smoother).

#### Posterior Predictive

$$\mu_* = K(X_*, X)\,(K + \sigma_n^2 I)^{-1} y$$

$$\text{var}_* = K(X_*, X_*) - K(X_*, X)\,(K + \sigma_n^2 I)^{-1} K(X, X_*)$$

#### Log Marginal Likelihood (model selection)

$$\log p(y \mid X) = -\tfrac{1}{2} y^\top (K + \sigma_n^2 I)^{-1} y - \tfrac{1}{2}\log|K + \sigma_n^2 I| - \tfrac{n}{2}\log 2\pi$$

#### Application

- Regression with uncertainty quantification.
- Bayesian optimisation (surrogate model).
- Time-series forecasting.
- Active learning.

#### Implementation Steps

1. Generate noisy training data: $y = \sin(x) + \epsilon$, $\epsilon \sim \mathcal{N}(0, 0.2^2)$.
2. Build kernel matrix $K = k(X, X)$ and add noise: $K_n = K + \sigma_n^2 I$.
3. Cholesky factorise: $K_n = LL^\top$.
4. Solve for $\alpha$: forward sub → $z = L^{-1}y$; backward sub → $\alpha = L^{-\top}z$.
5. Compute log marginal likelihood from $L$ and $\alpha$.
6. At test points $X_*$: compute $K_* = k(X, X_*)$; mean $= K_*^\top \alpha$.
7. Solve $L V = K_*$ (forward sub per column); variance $= \text{diag}(K_{**}) - \sum V^2$.
8. Plot mean ± 2σ confidence band; save figure.
9. Return `(x, y, x_test, mean, variance)`.

#### Usage

```python
from applications.gp_regression import gp_regression

x, y, x_test, mean, variance = gp_regression(
    n_samples=50, sigma=2.0, length_scale=0.5, sigma_n=1.0
)
```

---

## 4. Running the Demo Notebook

The notebook [`notebooks/demo_engine.ipynb`](notebooks/demo_engine.ipynb) is a **Restart & Run All** walkthrough of all six module groups with inline plots.

```bash
# From the repository root
$env:PYTHONPATH = "src"          # Windows PowerShell
export PYTHONPATH=src            # macOS / Linux

jupyter notebook notebooks/demo_engine.ipynb
```

Then in Jupyter: **Kernel → Restart & Run All**.

| Section | Content |
|---------|---------|
| 1 — LU Decomposition | P, L, U printed; residual; L/U heatmaps |
| 2 — Cholesky & QR | Factor matrices; orthogonality check; heatmaps |
| 3 — Linear Solvers | Triangular solve verification; CG convergence |
| 4 — Least Squares | Fit vs true line; scatter + fitted line plot |
| 5 — SVD Compression | Singular value spectrum; 4-panel image grid |
| 6 — Gaussian Process | GP posterior plot; length-scale comparison |

---

## 5. Running Tests

```bash
# From the repository root
$env:PYTHONPATH = "src"          # Windows PowerShell
export PYTHONPATH=src            # macOS / Linux

pytest                           # run all 46 tests
pytest tests/test_decompositions.py   # single module
pytest -v                        # verbose output
```

Expected result: **46 passed**.
