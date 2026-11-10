"""
path_generators.py

Path generation methods including dimension reduction techniques.

Pure constructors (take Z as input):
  - 'standard' : sequential Brownian motion (baseline)
  - 'bb'       : Brownian Bridge (Morokoff & Caflisch 1994)
  - 'pca'      : PCA (Acworth et al. 1998)
  - 'gpca'     : Gradient-based PCA (Xiao & Wang 2017, 2018)

Discontinuity alignment:
  - construct_with_ot : applies DiscontinuityOT from discontinuity_treatment.py

CQR dimension reduction (Weng, Wang & He 2017):
  - CQRDimensionReduction : k-means clustering + QR decomposition on the
    smoothed integrand.

"""

import warnings
import numpy as np


class PathGenerator:
    """Generate Brownian paths using various methods."""

    # ============================================================
    # PURE CONSTRUCTORS (take Z as input)
    # ============================================================

    @staticmethod
    def construct_standard(S0: float, drift: float, diffusion: float,
                           Z: np.ndarray) -> np.ndarray:
        """Sequential Brownian motion (baseline)."""
        log_returns = drift + diffusion * Z
        log_paths = np.cumsum(log_returns, axis=1)
        return S0 * np.exp(log_paths)

    @staticmethod
    def construct_bb(S0: float, drift: float, diffusion: float,
                     Z: np.ndarray, T: float) -> np.ndarray:
        """Brownian Bridge construction (Morokoff & Caflisch 1994)."""
        n_sims, n_steps = Z.shape
        dt = T / n_steps
        r_eff = drift / dt
        sigma_eff = diffusion / np.sqrt(dt)

        is_pow2 = (n_steps & (n_steps - 1)) == 0 and n_steps > 0
        if not is_pow2:
            raise ValueError(
                f"construct_bb requires n_steps to be a power of 2; "
                f"got {n_steps}."
            )

        times = np.arange(1, n_steps + 1) * dt
        W = np.zeros((n_sims, n_steps + 1))
        W[:, n_steps] = np.sqrt(T) * Z[:, 0]

        idx = 1
        stride = n_steps // 2
        while stride >= 1:
            for i in range(stride, n_steps, 2 * stride):
                left = i - stride
                right = i + stride
                t_l = left * dt
                t_r = right * dt
                t_i = i * dt
                a = (t_r - t_i) / (t_r - t_l)
                b = (t_i - t_l) / (t_r - t_l)
                var = (t_i - t_l) * (t_r - t_i) / (t_r - t_l)
                if var > 0 and idx < n_steps:
                    W[:, i] = (a * W[:, left] + b * W[:, right]
                               + np.sqrt(var) * Z[:, idx])
                    idx += 1
            stride //= 2

        W_levels = W[:, 1:]
        log_paths = (np.log(S0)
                     + r_eff * times[None, :]
                     + sigma_eff * W_levels)
        return np.exp(log_paths)

    @staticmethod
    def construct_pca(S0: float, drift: float, diffusion: float,
                      Z: np.ndarray, T: float) -> np.ndarray:
        """PCA-based construction (Acworth et al. 1998)."""
        n_sims, n_steps = Z.shape
        dt = T / n_steps
        r_eff = drift / dt
        sigma_eff = diffusion / np.sqrt(dt)
        times = np.arange(1, n_steps + 1) * dt

        covariance = np.minimum.outer(times, times)
        eigenvals, eigenvecs = np.linalg.eigh(covariance)
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]

        for k in range(eigenvecs.shape[1]):
            j = int(np.argmax(np.abs(eigenvecs[:, k])))
            if eigenvecs[j, k] < 0:
                eigenvecs[:, k] = -eigenvecs[:, k]

        # FIX: comment kept but line clarified
        W_levels = Z @ (np.sqrt(eigenvals)[:, None] * eigenvecs.T)

        log_paths = (np.log(S0)
                     + r_eff * times[None, :]
                     + sigma_eff * W_levels)
        return np.exp(log_paths)

    # ============================================================
    # GRADIENT-BASED PCA (Xiao & Wang 2017, 2018)
    # ============================================================
    @staticmethod
    def _gradient_complex_step(grad_fn, Z_grad, eps):
        """Complex-step gradient. Requires complex-safe grad_fn."""
        m, d = Z_grad.shape
        Z_c = Z_grad.astype(complex)
        D = np.zeros((m, d))
        for k in range(d):
            Z_probe = Z_c.copy()
            Z_probe[:, k] += 1j * eps
            D[:, k] = np.imag(grad_fn(Z_probe)) / eps
        return D

    @staticmethod
    def _gradient_central_diff(grad_fn, Z_grad, eps):
        """Central finite-difference fallback."""
        m, d = Z_grad.shape
        D = np.zeros((m, d))
        for k in range(d):
            Zp = Z_grad.copy(); Zp[:, k] += eps
            Zm = Z_grad.copy(); Zm[:, k] -= eps
            D[:, k] = np.real(grad_fn(Zp) - grad_fn(Zm)) / (2.0 * eps)
        return D

    @staticmethod
    def construct_gpca(S0, drift, diffusion, Z, T,
                       grad_fn,
                       n_grad_samples=8192,
                       eps=1e-6,
                       Z_grad=None,
                       A_cond=None,
                       t1=None,
                       truncate_dims=None,
                       return_basis=False):
        """
        Gradient-based PCA (Xiao & Wang 2017, 2018).

        Removed: level-space coordinate conversion. That transformation
        was mathematically invalid because the inverse cumulative-sum
        map is not orthogonal, and it produced non-orthogonal Q_inc
        that did not preserve the distribution of Z. The correct GPCA
        operates in the original coordinate space.
        """
        n_sims, n_steps = Z.shape
        d = n_steps
        dt = T / n_steps
        r_eff = drift / dt
        sigma_eff = diffusion / np.sqrt(dt)

        paper_mode = (A_cond is not None) and (t1 is not None)
        if (A_cond is None) != (t1 is None):
            raise ValueError(
                "A_cond and t1 must be provided together or both omitted."
            )

        if truncate_dims is not None:
            warnings.warn(
                "truncate_dims is a non-paper extension; use with caution.",
                stacklevel=2,
            )

        if paper_mode:
            A_cond = np.asarray(A_cond)
            if A_cond.shape != (d - 1, d - 1):
                raise ValueError(
                    f"A_cond must have shape ({d-1}, {d-1}), got "
                    f"{A_cond.shape}."
                )
            gpca_dim = d - 1
        else:
            gpca_dim = d

        # 1. Sample points for gradient estimation
        if Z_grad is None:
            try:
                from scipy.stats import qmc, norm
                sobol = qmc.Sobol(d=gpca_dim, scramble=True, seed=0)
                U = sobol.random(n_grad_samples)
                U = np.clip(U, 1e-10, 1 - 1e-10)
                Z_grad = norm.ppf(U)
            except ImportError:
                rng = np.random.default_rng(0)
                Z_grad = rng.standard_normal((n_grad_samples, gpca_dim))

        if Z_grad.shape[1] != gpca_dim:
            raise ValueError(
                f"Z_grad has {Z_grad.shape[1]} columns, expected {gpca_dim}."
            )

        # 2. Gradient matrix (complex step with fallback)
        try:
            D = PathGenerator._gradient_complex_step(grad_fn, Z_grad, eps)
            if not np.all(np.isfinite(D)):
                raise ValueError("complex-step produced non-finite values")
        except (TypeError, ValueError, FloatingPointError):
            D = PathGenerator._gradient_central_diff(grad_fn, Z_grad, eps)

        if not np.all(np.isfinite(D)):
            raise RuntimeError(
                "GPCA: gradient matrix has non-finite entries. Check "
                "grad_fn for domain errors."
            )

        # 2b. Raw-payoff misuse warning
        frac_near_zero = np.mean(np.abs(D) < 1e-12)
        if frac_near_zero > 0.999:
            warnings.warn(
                f"GPCA: {100*frac_near_zero:.1f}% of gradient entries "
                f"are near-zero. grad_fn may be returning the RAW "
                f"discontinuous payoff rather than a smoothed integrand.",
                stacklevel=2,
            )

        # 3. Gradient covariance
        C = D.T @ D

        # 4. Eigendecomposition
        eigenvals, eigenvecs = np.linalg.eigh(C)
        order = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[order]
        eigenvecs = eigenvecs[:, order]

        # Clamp tiny negative eigenvalues to zero (numerical round-off)
        eigenvals = np.maximum(eigenvals, 0.0)

        # Deterministic sign convention
        for j in range(eigenvecs.shape[1]):
            imax = int(np.argmax(np.abs(eigenvecs[:, j])))
            if eigenvecs[imax, j] < 0:
                eigenvecs[:, j] = -eigenvecs[:, j]

        Q = eigenvecs

        # 4b. Effective-dimension diagnostic
        total = float(eigenvals.sum())
        if total > 0:
            cum = np.cumsum(eigenvals) / total
            k99 = int(np.searchsorted(cum, 0.99) + 1)
            if k99 > max(4, gpca_dim // 2):
                warnings.warn(
                    f"GPCA: 99% of gradient variance requires "
                    f"{k99} of {gpca_dim} dimensions. The integrand "
                    f"has high effective dimension; low VRF may be "
                    f"intrinsic.",
                    stacklevel=2,
                )

        # 5. Transform and reconstruct
        if paper_mode:
            Z1 = Z[:, 0]
            Z_rest = Z[:, 1:]
            Z_rest_gpca = Z_rest @ Q.T

            if truncate_dims is not None:
                if not (1 <= truncate_dims <= gpca_dim):
                    raise ValueError(
                        f"truncate_dims must be in [1, {gpca_dim}]."
                    )
                Z_rest_gpca[:, truncate_dims:] = 0.0

            W1 = np.sqrt(t1) * Z1
            dW = Z_rest_gpca @ A_cond.T

            W = np.zeros((n_sims, d))
            W[:, 0] = W1
            W[:, 1:] = W1[:, None] + dW

            times = np.arange(1, d + 1) * dt
            log_paths = (np.log(S0)
                         + r_eff * times[None, :]
                         + sigma_eff * W)
            paths = np.exp(log_paths)
            Q_applied = Q
        else:
            Z_gpca = Z @ Q.T

            if truncate_dims is not None:
                if not (1 <= truncate_dims <= gpca_dim):
                    raise ValueError(
                        f"truncate_dims must be in [1, {gpca_dim}]."
                    )
                Z_gpca[:, truncate_dims:] = 0.0

            log_returns = drift + diffusion * Z_gpca
            log_paths = np.cumsum(log_returns, axis=1)
            paths = S0 * np.exp(log_paths)
            Q_applied = Q

        if return_basis:
            return paths, (Q_applied, eigenvals)
        return paths
    # ============================================================
    # UNIFIED PURE INTERFACE
    # ============================================================

    @staticmethod
    def generate_paths_pure(path_constructor: str, S0: float, drift: float,
                            diffusion: float, Z: np.ndarray, T: float,
                            grad_fn: callable = None,
                            n_grad_samples: int = 8192,
                            eps: float = 1e-6,
                            Z_grad: np.ndarray = None,
                            A_cond: np.ndarray = None,
                            t1: float = None,
                            truncate_dims: int = None,
                            level_space: bool = True
                            ) -> np.ndarray:
        """Unified interface for pure path construction."""
        if path_constructor == 'standard':
            return PathGenerator.construct_standard(S0, drift, diffusion, Z)
        elif path_constructor == 'bb':
            return PathGenerator.construct_bb(S0, drift, diffusion, Z, T)
        elif path_constructor == 'pca':
            return PathGenerator.construct_pca(S0, drift, diffusion, Z, T)
        elif path_constructor == 'gpca':
            if grad_fn is None:
                raise ValueError("path_constructor='gpca' requires grad_fn.")
            return PathGenerator.construct_gpca(
                S0, drift, diffusion, Z, T,
                grad_fn=grad_fn,
                n_grad_samples=n_grad_samples,
                eps=eps,
                Z_grad=Z_grad,
                A_cond=A_cond,
                t1=t1,
                truncate_dims=truncate_dims,
                level_space=level_space,
            )
        else:
            raise ValueError(f"Unknown path_constructor: '{path_constructor}'.")

    @staticmethod
    def construct_with_ot(ot, S0: float, drift: float, diffusion: float,
                          Z: np.ndarray) -> np.ndarray:
        """Apply OT then standard construction. OT is construction-specific."""
        Z_ot = ot.transform(Z)
        return PathGenerator.construct_standard(S0, drift, diffusion, Z_ot)


# ================================================================
# CQR DIMENSION REDUCTION (Weng, Wang & He 2017)
# ================================================================
class CQRDimensionReduction:
    """CQR dimension reduction for smoothed integrands."""

    def __init__(self, n_clusters: int = 32,
                 random_state: int = 0,
                 max_iter: int = 300,
                 n_init: int = 10,
                 G_weight: float = 1.0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.G_weight = G_weight
        self.Q_ = None
        self.centroids_ = None
        self._n_features = None

    def fit(self, G_fn: callable, Z_sample: np.ndarray):
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise ImportError(
                "CQRDimensionReduction requires scikit-learn for k-means."
            ) from exc

        Z_sample = np.asarray(Z_sample, dtype=np.float64)
        if Z_sample.ndim != 2:
            raise ValueError(f"Z_sample must be 2-D, got {Z_sample.shape}.")
        m, d = Z_sample.shape
        if self.n_clusters < 1 or self.n_clusters > m:
            raise ValueError(
                f"n_clusters must be in [1, {m}], got {self.n_clusters}."
            )

        G_vals = np.asarray(G_fn(Z_sample), dtype=np.float64).reshape(-1)
        if G_vals.shape[0] != m:
            raise ValueError(
                f"G_fn returned {G_vals.shape[0]} values for {m} samples."
            )

        Z_std = (Z_sample - Z_sample.mean(axis=0)) / (
            Z_sample.std(axis=0) + 1e-12
        )
        G_std = (G_vals - G_vals.mean()) / (G_vals.std() + 1e-12)

        if self.G_weight > 0.0:
            joint = np.column_stack([Z_std, self.G_weight * G_std[:, None]])
        else:
            joint = Z_std

        self._n_features = joint.shape[1]

        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=self.n_init,
        )
        kmeans.fit(joint)
        joint_centroids = kmeans.cluster_centers_

        if self.G_weight > 0.0:
            centroid_Z = joint_centroids[:, :d]
        else:
            centroid_Z = joint_centroids

        Q_partial, _ = np.linalg.qr(centroid_Z.T)
        k = Q_partial.shape[1]

        if k < d:
            rng = np.random.default_rng(self.random_state)
            extra = rng.standard_normal((d, d - k))
            combined = np.hstack([Q_partial, extra])
            Q_full, _ = np.linalg.qr(combined)
        else:
            Q_full = Q_partial

        for j in range(Q_full.shape[1]):
            imax = int(np.argmax(np.abs(Q_full[:, j])))
            if Q_full[imax, j] < 0:
                Q_full[:, j] = -Q_full[:, j]

        self.Q_ = Q_full
        self.centroids_ = centroid_Z
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        if self.Q_ is None:
            raise RuntimeError("CQRDimensionReduction.transform before fit.")
        Z = np.asarray(Z, dtype=np.float64)
        if Z.ndim != 2:
            raise ValueError(f"Z must be 2-D, got shape {Z.shape}.")
        if Z.shape[1] != self.Q_.shape[0]:
            raise ValueError(
                f"Z has {Z.shape[1]} columns but CQR basis expects "
                f"{self.Q_.shape[0]}."
            )
        return Z @ self.Q_

    def fit_transform(self, G_fn: callable, Z_sample: np.ndarray,
                      Z_to_transform: np.ndarray = None) -> np.ndarray:
        self.fit(G_fn, Z_sample)
        if Z_to_transform is None:
            Z_to_transform = Z_sample
        return self.transform(Z_to_transform)