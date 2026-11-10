"""
discontinuity_treatment.py

Payoff-aware Orthogonal Transformation (OT) method for discontinuous payoffs

References
----------
- Wang, X. and Tan, K. S. (2013). "Pricing and Hedging with Discontinuous
  Functions: Quasi-Monte Carlo Methods and Dimension Reduction."
  Management Science 59(2):376-389.                        [OT]

Key idea
--------
For a payoff of the form
    g(S) = f(S) * 1{b(w^T Z) >= 0}
where b is the discontinuity boundary, the OT method constructs an
orthogonal matrix U whose first column is w / ||w||. This aligns the
discontinuity with the first coordinate axis, making it QMC-friendly.

The vector w is derived ANALYTICALLY from the payoff structure, NOT
numerically estimated. 

"""

import warnings
import numpy as np
from typing import Optional, Dict, Callable

from path_generators import PathGenerator

# ----------------------------------------------------------------------
# Supported option types
# ----------------------------------------------------------------------

SUPPORTED_OPTION_TYPES = {
    'digital_european',
    'digital_geometric_asian',
    'knockout_geometric_asian',
    'standard_barrier',
}

EXACT_DIRECTION_TYPES = {
    'digital_european',
    'digital_geometric_asian',
    'knockout_geometric_asian',
}

APPROXIMATE_DIRECTION_TYPES = {'standard_barrier'}


class DiscontinuityOT:
    """
    Payoff-aware Orthogonal Transformation for discontinuous payoffs.

    The OT matrix U is built such that U[:, 0] = w_unit, where w_unit is
    the unit discontinuity direction in Z-space (the standard normal
    input to the path construction).
    """

    def __init__(self, option_type: str, n_steps: int, T: float,
                 sigma: float,
                 geometric_weights: Optional[np.ndarray] = None,
                 arithmetic_weights: Optional[np.ndarray] = None,
                 use_terminal_only: bool = False,
                 suppress_warnings: bool = False):
        self.option_type = option_type
        self.n_steps = n_steps
        self.T = T
        self.sigma = sigma
        self.use_terminal_only = use_terminal_only
        self.suppress_warnings = suppress_warnings

        if option_type not in SUPPORTED_OPTION_TYPES:
            raise ValueError(
                f"option_type='{option_type}' not supported. "
                f"Choose from {sorted(SUPPORTED_OPTION_TYPES)}."
            )

        if geometric_weights is None:
            geometric_weights = np.ones(n_steps) / n_steps
        else:
            geometric_weights = np.asarray(geometric_weights, dtype=float)
            if geometric_weights.shape != (n_steps,):
                raise ValueError(
                    f"geometric_weights must have shape ({n_steps},), "
                    f"got {geometric_weights.shape}."
                )
            if not np.isclose(geometric_weights.sum(), 1.0, atol=1e-10):
                raise ValueError(
                    f"geometric_weights must sum to 1, "
                    f"got sum={geometric_weights.sum()}."
                )
        self.geometric_weights = geometric_weights

        if arithmetic_weights is None:
            arithmetic_weights = geometric_weights
        else:
            arithmetic_weights = np.asarray(arithmetic_weights, dtype=float)
            if arithmetic_weights.shape != (n_steps,):
                raise ValueError(
                    f"arithmetic_weights must have shape ({n_steps},), "
                    f"got {arithmetic_weights.shape}."
                )
            if not np.isclose(arithmetic_weights.sum(), 1.0, atol=1e-10):
                raise ValueError(
                    f"arithmetic_weights must sum to 1, "
                    f"got sum={arithmetic_weights.sum()}."
                )
        self.arithmetic_weights = arithmetic_weights

        if (option_type in APPROXIMATE_DIRECTION_TYPES
                and not suppress_warnings):
            warnings.warn(
                "standard_barrier: no analytic discontinuity direction "
                "exists in the Wang-Tan framework. Using a Taylor "
                "linearization heuristic.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------
    # Discontinuity direction
    # ------------------------------------------------------------------

    def discontinuity_direction(self) -> np.ndarray:
        """
        Return the unit discontinuity direction in Z-space (the standard
        normal input). This is the vector w such that the discontinuity
        boundary is a function of w^T Z.
        """
        if self.option_type == 'digital_european':
            return self._direction_digital_european()
        elif self.option_type in ('digital_geometric_asian',
                                   'knockout_geometric_asian'):
            return self._direction_geometric_asian()
        elif self.option_type == 'standard_barrier':
            return self._direction_standard_barrier()
        else:
            raise ValueError(f"Unhandled option_type: {self.option_type}")

    def _direction_digital_european(self) -> np.ndarray:
        """
        Digital European: 1{S_T >= K}.

        S_T = S_0 * exp(drift*n + diffusion*sum_i Z_i), so the boundary
        is sum_i Z_i >= c, i.e., the diagonal direction 1/sqrt(n).
        """
        if self.use_terminal_only:
            # S_T depends only on Z_1 in the conditional setting; use e_1.
            w = np.zeros(self.n_steps)
            w[0] = 1.0
        else:
            w = np.ones(self.n_steps) / np.sqrt(self.n_steps)
        return w

    def _direction_geometric_asian(self) -> np.ndarray:
        """
        Geometric Asian discontinuity direction (Examples B1 and B2).

        For the standard construction,
            log S_G = const + diffusion * nu^T Z,
        with nu_j = sum_{k >= j} w_k. The discontinuity boundary
        1{S_G < H} is a function of nu^T Z, so the unit direction is
        nu / ||nu||.
        """
        nu = self._geometric_asian_weight_vector()
        norm = np.linalg.norm(nu)
        if norm < 1e-15:
            raise ValueError("Geometric Asian weight vector has zero norm.")
        return nu / norm

    def _direction_standard_barrier(self) -> np.ndarray:
        """
        Standard barrier: (S_T - K)^+ * 1{min_t S_t > H}.

        No analytic direction exists in the Wang-Tan framework. OT cannot
        be applied to this payoff class.
        """
        raise NotImplementedError(
            "OT has no analytic discontinuity direction for the running-"
            "minimum barrier payoff. Use Auto-Realignment (Wang 2016) or "
            "QR (He & Wang 2014) instead, or use ConditionalQMC to smooth "
            "the discontinuity analytically."
        )

    def _geometric_asian_weight_vector(self) -> np.ndarray:
        """
        Cumulative weight vector nu in Z-space:
        nu_j = sum_{k=j}^{n-1} w_k, using geometric_weights.
        """
        w = self.geometric_weights
        nu = np.array([np.sum(w[j:]) for j in range(self.n_steps)])
        return nu

    # ------------------------------------------------------------------
    # Orthogonal matrix construction
    # ------------------------------------------------------------------

    def build_ot_matrix(self) -> np.ndarray:
        """
        Build orthogonal U with U[:, 0] = w_unit, using a Householder
        reflection H that maps w_unit to e_1. H is symmetric and
        orthogonal, so U = H works directly.
        """
        w = self.discontinuity_direction()
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-15:
            raise ValueError("Discontinuity direction has zero norm.")
        w_unit = w / w_norm

        e1 = np.zeros(self.n_steps)
        e1[0] = 1.0

        if np.allclose(w_unit, e1, atol=1e-14):
            return np.eye(self.n_steps)

        if np.allclose(w_unit, -e1, atol=1e-14):
            U = np.eye(self.n_steps)
            U[0, 0] = -1.0
            return U

        u = w_unit - e1
        u_norm = np.linalg.norm(u)
        if u_norm < 1e-15:
            return np.eye(self.n_steps)
        u = u / u_norm
        H = np.eye(self.n_steps) - 2.0 * np.outer(u, u)

        if not np.allclose(H.T @ H, np.eye(self.n_steps), atol=1e-12):
            raise RuntimeError("OT matrix is not orthogonal.")
        if not np.allclose(H[:, 0], w_unit, atol=1e-12):
            raise RuntimeError(
                "First column of OT matrix is not the discontinuity "
                "direction."
            )
        return H

    # ------------------------------------------------------------------
    # Transform (pure rotation, distribution-preserving)
    # ------------------------------------------------------------------

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply OT: Z_ot = Z @ U.T (row-vector convention).

        U is orthogonal, so Z_ot has the same distribution as Z (standard
        normal, i.i.d.). The discontinuity in the payoff, which was a
        function of w_unit^T Z in raw coordinates, becomes a function of
        Z_ot[:, 0] (the first coordinate) in rotated coordinates. This is
        what makes QMC effective.
        """
        if Z.ndim != 2 or Z.shape[1] != self.n_steps:
            raise ValueError(
                f"Z must have shape (n_simulations, {self.n_steps}), "
                f"got {Z.shape}."
            )
        U = self.build_ot_matrix()
        return Z @ U.T

    # ------------------------------------------------------------------
    # Payoff construction
    # ------------------------------------------------------------------

    def payoff(self, Z_ot: np.ndarray,
               S0: float, drift: float, diffusion: float,
               K: float, H: Optional[float] = None) -> np.ndarray:
        """
        Compute the paper's payoff on OT-transformed samples.

        Z_ot is the output of `transform`. The path is constructed via
        construct_standard(Z_ot).
        """
        if Z_ot.ndim != 2 or Z_ot.shape[1] != self.n_steps:
            raise ValueError(
                f"Z_ot must have shape (n_simulations, {self.n_steps}), "
                f"got {Z_ot.shape}."
            )

        paths = PathGenerator.construct_standard(
            S0, drift, diffusion, Z_ot
        )

        if self.option_type == 'digital_european':
            ST = paths[:, -1]
            return (ST >= K).astype(float)

        if self.option_type in ('digital_geometric_asian',
                                 'knockout_geometric_asian'):
            if H is None:
                raise ValueError(
                    f"H (geometric barrier/threshold) is required for "
                    f"option_type='{self.option_type}'."
                )
            S_G = self._geometric_average(paths)
            indicator = (S_G < H).astype(float)

            if self.option_type == 'digital_geometric_asian':
                return indicator

            S_A = self._arithmetic_average(paths)
            return np.maximum(S_A - K, 0.0) * indicator

        if self.option_type == 'standard_barrier':
            if H is None:
                raise ValueError(
                    "H (barrier) is required for 'standard_barrier'."
                )
            ST = paths[:, -1]
            running_min = paths.min(axis=1)
            return np.maximum(ST - K, 0.0) * (running_min > H).astype(float)

        raise ValueError(f"Unhandled option_type: {self.option_type}")

    def _geometric_average(self, paths: np.ndarray) -> np.ndarray:
        log_paths = np.log(paths)
        return np.exp(log_paths @ self.geometric_weights)

    def _arithmetic_average(self, paths: np.ndarray) -> np.ndarray:
        return paths @ self.arithmetic_weights

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def is_exact(self) -> bool:
        return self.option_type in EXACT_DIRECTION_TYPES

    def check_orthogonality(self, atol: float = 1e-12) -> bool:
        U = self.build_ot_matrix()
        return bool(np.allclose(U.T @ U, np.eye(self.n_steps), atol=atol))

    def check_alignment(self, atol: float = 1e-12) -> bool:
        U = self.build_ot_matrix()
        w = self.discontinuity_direction()
        w_unit = w / np.linalg.norm(w)
        return bool(np.allclose(U[:, 0], w_unit, atol=atol))

    def effective_dimension(self, Z: np.ndarray,
                            payoff_fn: Callable[[np.ndarray], np.ndarray],
                            threshold: float = 0.99) -> Dict:
        Z_ot = self.transform(Z)
        payoffs_full = payoff_fn(Z_ot)
        var_full = float(np.var(payoffs_full))

        if var_full <= 1e-30:
            return {
                'truncation_dim': 0,
                'variance_ratio': np.zeros(self.n_steps),
                'per_dim_var': np.zeros(self.n_steps),
                'dimension_order': np.arange(self.n_steps),
            }

        per_dim_var = np.zeros(self.n_steps)
        for j in range(self.n_steps):
            Z_fixed = Z_ot.copy()
            Z_fixed[:, j] = 0.0
            payoffs_fixed = payoff_fn(Z_fixed)
            per_dim_var[j] = var_full - float(np.var(payoffs_fixed))

        order = np.argsort(per_dim_var)[::-1]
        cumvar = np.cumsum(per_dim_var[order]) / var_full
        truncation_dim = int(np.searchsorted(cumvar, threshold) + 1)

        return {
            'truncation_dim': truncation_dim,
            'variance_ratio': cumvar,
            'per_dim_var': per_dim_var,
            'dimension_order': order,
        }


# ======================================================================
# Convenience wrapper
# ======================================================================

def generate_paths_with_ot(ot: DiscontinuityOT,
                           path_constructor: str,
                           S0: float, drift: float, diffusion: float,
                           Z: np.ndarray) -> np.ndarray:
    if path_constructor != 'standard':
        raise ValueError(
            f"OT is only supported with path_constructor='standard'. "
            f"Got '{path_constructor}'."
        )
    Z_ot = ot.transform(Z)
    return PathGenerator.construct_standard(S0, drift, diffusion, Z_ot)


# ======================================================================
# Integrated Linear Transformation (Imai & Tan 2014)
# ======================================================================

class DiscontinuityILT(DiscontinuityOT):
    """
    Integrated Linear Transformation (ILT) method of Imai & Tan (2014).

    Extends Wang & Tan (2013) OT by adding dimension-reduction columns
    to the orthogonal matrix. The first column handles the discontinuity
    (as in OT); the remaining columns optimally reduce the truncation
    dimension of the continuous part of the payoff.

    NOTE ON ALGORITHMIC APPROXIMATION:
    The original ILT method constructs the dimension-reduction columns
    by solving an optimization problem derived from a Taylor expansion
    of the payoff. The implementation here approximates those columns
    by the leading eigenvectors of a sampled gradient covariance matrix,
    which is a numerically tractable surrogate for the analytical
    construction.

    Reference
    ---------
    Imai, J. and Tan, K. S. (2014). "Pricing Derivative Securities Using
    Integrated Quasi-Monte Carlo Methods with Dimension Reduction and
    Discontinuity Realignment." SIAM J. Sci. Comput., 36(5):A2101-A2121.
    """

    def __init__(self, option_type: str, n_steps: int, T: float,
                 sigma: float,
                 geometric_weights: Optional[np.ndarray] = None,
                 arithmetic_weights: Optional[np.ndarray] = None,
                 n_grad_samples: int = 8192,
                 eps: float = 1e-6,
                 grad_fn: Optional[Callable] = None,
                 seed: int = 0,
                 use_sampled_covariance: bool = False,
                 suppress_warnings: bool = False):
        super().__init__(
            option_type=option_type,
            n_steps=n_steps,
            T=T,
            sigma=sigma,
            geometric_weights=geometric_weights,
            arithmetic_weights=arithmetic_weights,
            use_terminal_only=False,
            suppress_warnings=suppress_warnings,
        )
        self.n_grad_samples = n_grad_samples
        self.eps = eps
        self.grad_fn = grad_fn
        self.seed = seed
        self.use_sampled_covariance = use_sampled_covariance

    def build_ilt_matrix(self) -> np.ndarray:
        """
        Build the ILT orthogonal matrix A*.

        Column 1: discontinuity direction (same as OT).
        Columns 2..d: dimension-reduction columns from the leading
        eigenvectors of the gradient covariance of f_2, orthogonalized
        against column 1.

        Uses SVD-based orthogonal completion to robustly handle
        rank-deficient gradient covariances.
        """
        d = self.n_steps
        w = self.discontinuity_direction()
        w_unit = w / np.linalg.norm(w)

        grad_cov = self._estimate_gradient_covariance()

        # Rank diagnostic with relative tolerance
        eigvals_cov = np.linalg.eigvalsh(grad_cov)
        eigvals_cov = np.maximum(eigvals_cov, 0.0)
        if eigvals_cov.max() <= 0:
            numerical_rank = 0
        else:
            tol = eigvals_cov.max() * 1e-6 * max(d, 1)
            numerical_rank = int(np.sum(eigvals_cov > tol))

        if numerical_rank <= 1 and not self.suppress_warnings:
            warnings.warn(
                f"ILT gradient covariance is numerically rank-{numerical_rank}. "
                f"The dimension-reduction columns will be dominated by the "
                f"single gradient direction. Consider increasing "
                f"n_grad_samples or supplying a grad_fn with a richer "
                f"directional structure.",
                stacklevel=2,
            )

        eigenvals, eigenvecs = np.linalg.eigh(grad_cov)
        order = eigenvals.argsort()[::-1]
        eigenvecs = eigenvecs[:, order]

        for j in range(eigenvecs.shape[1]):
            imax = int(np.argmax(np.abs(eigenvecs[:, j])))
            if eigenvecs[imax, j] < 0:
                eigenvecs[:, j] = -eigenvecs[:, j]

        # Ordered orthogonal completion: preserve leading projected gradient
        # directions. An SVD of all candidates rotates this basis and may
        # destroy the intended QMC importance ordering.
        columns = [w_unit]
        tolerance = 1e-10
        for j in range(d):
            v = eigenvecs[:, j].copy()
            for q in columns:
                v -= np.dot(q, v) * q
            length = np.linalg.norm(v)
            if length > tolerance:
                columns.append(v / length)
            if len(columns) == d:
                break
        # Deterministic completion if gradient covariance is rank deficient.
        for j in range(d):
            if len(columns) == d:
                break
            v = np.eye(d)[:, j].copy()
            for q in columns:
                v -= np.dot(q, v) * q
            length = np.linalg.norm(v)
            if length > tolerance:
                columns.append(v / length)
        Q_rest = np.column_stack(columns[1:])

        A = np.column_stack([w_unit, Q_rest])

        # Looser tolerance for the final orthogonality check
        if not np.allclose(A.T @ A, np.eye(d), atol=1e-8):
            raise RuntimeError(
                f"ILT matrix is not orthogonal (max deviation "
                f"{np.max(np.abs(A.T @ A - np.eye(d))):.2e})."
            )
        if not np.allclose(A[:, 0], w_unit, atol=1e-12):
            raise RuntimeError(
                "ILT first column is not the discontinuity direction."
            )
        return A

    def _default_grad_fn(self) -> Callable[[np.ndarray], np.ndarray]:
        option_type = self.option_type
        sigma = self.sigma
        geom_w = self.geometric_weights
        arith_w = self.arithmetic_weights

        def grad_fn(Z_ot: np.ndarray) -> np.ndarray:
            Z_ot = np.asarray(Z_ot)
            paths = PathGenerator.construct_standard(
                S0=1.0, drift=0.0, diffusion=sigma, Z=Z_ot
            )
            if option_type == 'knockout_geometric_asian':
                S_A = paths @ arith_w
                return np.maximum(S_A - 1.0, 0.0)
            if option_type == 'digital_geometric_asian':
                log_paths = np.log(paths)
                return np.exp(log_paths @ geom_w)
            if option_type == 'digital_european':
                return paths[:, -1]
            raise ValueError(
                f"ILT default grad_fn not defined for {option_type}."
            )

        return grad_fn

    def _estimate_gradient_covariance(self) -> np.ndarray:
        if self.use_sampled_covariance:
            return self._gradient_covariance_fd()

        d = self.n_steps
        if self.option_type == 'knockout_geometric_asian':
            grad_dir = np.cumsum(self.arithmetic_weights[::-1])[::-1].astype(float)
        elif self.option_type == 'digital_european':
            grad_dir = np.ones(d) / np.sqrt(d)
        else:
            grad_dir = self._geometric_asian_weight_vector().astype(float)
        norm = np.linalg.norm(grad_dir)
        if norm < 1e-15:
            raise ValueError("Gradient direction has zero norm.")
        grad_dir = grad_dir / norm
        return np.outer(grad_dir, grad_dir)

    def _gradient_covariance_fd(self) -> np.ndarray:
        d = self.n_steps

        if self.grad_fn is None:
            self.grad_fn = self._default_grad_fn()

        rng = np.random.default_rng(self.seed)
        Z_grad = rng.standard_normal((self.n_grad_samples, d))

        probe = np.asarray(self.grad_fn(Z_grad)).reshape(-1)
        if probe.shape[0] != self.n_grad_samples:
            raise ValueError(
                f"grad_fn returned {probe.shape[0]} values for "
                f"{self.n_grad_samples} samples."
            )
        if np.std(probe) < 1e-12:
            warnings.warn(
                "grad_fn output has near-zero variance across samples.",
                stacklevel=2,
            )

        D = np.zeros((self.n_grad_samples, d))
        for k in range(d):
            Z_plus = Z_grad.copy()
            Z_minus = Z_grad.copy()
            Z_plus[:, k] += self.eps
            Z_minus[:, k] -= self.eps
            D[:, k] = (self.grad_fn(Z_plus)
                       - self.grad_fn(Z_minus)) / (2.0 * self.eps)

        return (D.T @ D) / self.n_grad_samples

    def transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Apply ILT: Z_ilt = Z @ A. Pure rotation, distribution-preserving.
        """
        if Z.ndim != 2 or Z.shape[1] != self.n_steps:
            raise ValueError(
                f"Z must have shape (n_simulations, {self.n_steps}), "
                f"got {Z.shape}."
            )
        A = self.build_ilt_matrix()
        return Z @ A.T

    def transform_ilt(self, Z: np.ndarray) -> np.ndarray:
        return self.transform(Z)


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    from scipy.stats import qmc, norm

    def draw_sobol_normals(n, d, seed):
        """d-dimensional scrambled Sobol' mapped to standard normals."""
        s = qmc.Sobol(d=d, scramble=True, seed=seed)
        U = s.random(n)
        return norm.ppf(np.clip(U, 1e-10, 1 - 1e-10))

    def draw_sobol_plus_mc(n, d_eff, d_full, seed):
        """
        Draw a d_eff-dimensional Sobol' net in the FIRST d_eff
        coordinates, and plain MC in the remaining d_full - d_eff.
        """
        Z_eff = draw_sobol_normals(n, d_eff, seed)
        if d_full > d_eff:
            rng = np.random.default_rng(seed + 1)
            Z_rest = rng.standard_normal((n, d_full - d_eff))
            return np.hstack([Z_eff, Z_rest])
        return Z_eff

    np.random.seed(0)
    n_steps = 16
    T = 1.0
    sigma = 0.2
    S0 = 100.0
    r = 0.05
    q = 0.0
    drift = r - q - 0.5 * sigma ** 2
    diffusion = sigma

    # ==================================================================
    # Structural invariants
    # ==================================================================
    print("=" * 60)
    print("Structural invariants")
    print("=" * 60)

    ot = DiscontinuityOT('digital_european', n_steps, T, sigma,
                         use_terminal_only=True)
    print("\n=== Digital European (terminal-only) ===")
    print("Direction (first 5):", ot.discontinuity_direction()[:5])
    print("U orthogonal:      ", ot.check_orthogonality())
    print("U[:,0] == w_unit:  ", ot.check_alignment())

    ot2 = DiscontinuityOT('digital_european', n_steps, T, sigma,
                          use_terminal_only=False)
    print("\n=== Digital European (full-dim standard) ===")
    print("Direction (first 5):", ot2.discontinuity_direction()[:5])
    print("U orthogonal:      ", ot2.check_orthogonality())
    print("U[:,0] == w_unit:  ", ot2.check_alignment())

    ot3 = DiscontinuityOT('digital_geometric_asian', n_steps, T, sigma)
    print("\n=== Digital Geometric Asian (Example B1) ===")
    print("Direction (first 5):", ot3.discontinuity_direction()[:5])
    print("is_exact:           ", ot3.is_exact())
    print("U orthogonal:      ", ot3.check_orthogonality())
    print("U[:,0] == w_unit:  ", ot3.check_alignment())

    ot4 = DiscontinuityOT('knockout_geometric_asian', n_steps, T, sigma)
    print("\n=== Knock-out Geometric Asian (Example B2) ===")
    print("Direction (first 5):", ot4.discontinuity_direction()[:5])
    print("Same as digital Asian:",
          np.allclose(ot3.discontinuity_direction(),
                      ot4.discontinuity_direction()))
    print("is_exact:           ", ot4.is_exact())

    print("\n=== Standard Barrier (should raise NotImplementedError) ===")
    ot5 = DiscontinuityOT('standard_barrier', n_steps, T, sigma,
                          suppress_warnings=True)
    try:
        ot5.discontinuity_direction()
        print("ERROR: did not raise")
    except NotImplementedError:
        print("Raised NotImplementedError as expected.")

    # ==================================================================
    # Distribution-preservation check (critical sanity test)
    # ==================================================================
    print("\n" + "=" * 60)
    print("Distribution preservation check")
    print("=" * 60)

    Z_test = np.random.default_rng(0).standard_normal((100000, n_steps))
    for label, ot_test in [
        ("OT ", DiscontinuityOT('digital_geometric_asian', n_steps, T, sigma)),
        ("ILT", DiscontinuityILT('knockout_geometric_asian', n_steps, T, sigma,
                                 use_sampled_covariance=True,
                                 n_grad_samples=2048)),
    ]:
        Z_rot = ot_test.transform(Z_test)
        mean_dev = np.max(np.abs(Z_rot.mean(axis=0)))
        var_dev = np.max(np.abs(Z_rot.var(axis=0) - 1.0))
        print(f"{label}: max |mean| = {mean_dev:.4f}  "
              f"max |var-1| = {var_dev:.4f}  "
              f"(both should be ~0 for distribution preservation)")

    # ==================================================================
    # Alignment diagnostic
    # ==================================================================
    print("\n" + "=" * 60)
    print("Alignment diagnostic (does coord 0 carry the discontinuity?)")
    print("=" * 60)

    Z = draw_sobol_normals(2**15, n_steps, 42)
    ot_inst = DiscontinuityOT('digital_geometric_asian', n_steps, T, sigma)
    Z_ot = ot_inst.transform(Z)
    ind = ot_inst.payoff(Z_ot, S0, drift, sigma, K=100.0, H=100.0)
    corrs = [abs(np.corrcoef(Z_ot[:, k], ind)[0, 1])
             for k in range(n_steps)]
    top3 = np.argsort(corrs)[::-1][:3]
    print(f"OT top-3 |corr| coords: {top3}, "
          f"|r| = {[round(corrs[k], 3) for k in top3]}")

    # ==================================================================
    # VRF table
    # ==================================================================
    print("\n" + "=" * 60)
    print("QMC variance reduction factors")
    print("=" * 60)

    def vrf_case(name, option_type, K, H,
                 n_steps=16, n_sims=2**15, seed=42, d_eff=1):
        """
        Compare plain QMC, OT-QMC, ILT-QMC.

        All three methods draw their samples from the SAME scrambled
        Sobol' net (first d_eff coordinates QMC, rest MC). The
        difference is the transform applied BEFORE the payoff is
        evaluated:

          - Plain QMC: no transform (raw Z)
          - OT-QMC   : Z_ot = Z @ U       (U from DiscontinuityOT)
          - ILT-QMC  : Z_ilt = Z @ A      (A from DiscontinuityILT)

        Both transforms are orthogonal, so the distribution of Z is
        preserved and the estimator remains unbiased.
        """
        S0, T, r, sigma = 100.0, 1.0, 0.05, 0.2
        drift = r - 0.5 * sigma ** 2
        arith = np.ones(n_steps) / n_steps

        # --- Single shared Sobol'+MC draw ---
        Z_raw = draw_sobol_plus_mc(n_sims, d_eff, n_steps, seed)

        # --- Plain QMC: no transform ---
        paths_plain = PathGenerator.construct_standard(S0, drift, sigma, Z_raw)
        SG_plain = np.exp(np.log(paths_plain) @ arith)
        SA_plain = paths_plain @ arith
        if option_type == 'digital_geometric_asian':
            pay_plain = (SG_plain < H).astype(float)
        else:
            pay_plain = np.maximum(SA_plain - K, 0.0) * (SG_plain < H)

        # --- OT-QMC: apply the OT matrix ---
        ot = DiscontinuityOT(option_type, n_steps, T, sigma,
                             geometric_weights=arith,
                             arithmetic_weights=arith)
        Z_ot = ot.transform(Z_raw)
        pay_ot = ot.payoff(Z_ot, S0, drift, sigma, K, H)

        # --- ILT-QMC: apply the ILT matrix ---
        ilt = DiscontinuityILT(option_type, n_steps, T, sigma,
                               geometric_weights=arith,
                               arithmetic_weights=arith,
                               use_sampled_covariance=True,
                               n_grad_samples=8192,
                               eps=1e-6,
                               seed=0,
                               suppress_warnings=True)
        Z_ilt = ilt.transform(Z_raw)
        pay_ilt = ilt.payoff(Z_ilt, S0, drift, sigma, K, H)

        v_plain = np.var(pay_plain)
        v_ot = np.var(pay_ot)
        v_ilt = np.var(pay_ilt)

        print(f"\n--- {name} (d_eff={d_eff}) ---")
        print(f"QMC       : var={v_plain:.4e}  mean={pay_plain.mean():.6f}")
        print(f"OT-QMC    : var={v_ot:.4e}  mean={pay_ot.mean():.6f}  "
              f"VRF={v_plain / v_ot:.2f}")
        print(f"ILT-QMC   : var={v_ilt:.4e}  mean={pay_ilt.mean():.6f}  "
              f"VRF={v_plain / v_ilt:.2f}")

        if option_type == 'knockout_geometric_asian':
            ind = (SG_plain < H).astype(float)
            cont = np.maximum(SA_plain - K, 0.0)
            var_from_disc = (np.var(ind) * (cont[ind == 1].mean() ** 2))
            frac = var_from_disc / max(np.var(pay_plain), 1e-30)
            print(f"Variance fraction from discontinuity: {frac:.2%}")

    vrf_case("Example B1 (digital geometric Asian, H=K=100)",
             'digital_geometric_asian', K=100.0, H=100.0, d_eff=1)
    vrf_case("Example B2, H=K=100",
             'knockout_geometric_asian', K=100.0, H=100.0, d_eff=3)
    vrf_case("Example B2, H=90, K=100",
             'knockout_geometric_asian', K=100.0, H=90.0, d_eff=3)

    print("\nAll self-tests complete.")
# ======================================================================
# General Wang--Tan (2013) OT: x=A0 U z, A0 A0.T = C.
# This API is independent of the legacy payoff-specific DiscontinuityOT.
# ======================================================================
class GeneralizedOT:
    """Exact axis alignment for a *single linear Gaussian boundary* q.T @ W.

    W denotes Brownian levels (not Brownian increments). The independent
    input Z is a column vector conceptually; batch row arrays use Z @ U.T
    @ A0.T. No exact alignment claim is made for a running-minimum barrier
    or an arithmetic-average threshold.
    """

    def __init__(self, A0: np.ndarray, q: np.ndarray):
        self.A0 = np.asarray(A0, dtype=float)
        self.q = np.asarray(q, dtype=float).reshape(-1)
        d = len(self.q)
        if self.A0.shape != (d, d):
            raise ValueError("A0 must be square and match q")
        if not np.isfinite(self.A0).all() or not np.isfinite(self.q).all():
            raise ValueError("A0 and q must be finite")
        w = self.A0.T @ self.q
        self.D = float(np.linalg.norm(w))
        if self.D <= 0:
            raise ValueError("The indicator direction must have positive variance")
        self.u1 = w / self.D
        e1 = np.eye(d)[:, 0]
        if np.linalg.norm(self.u1 - e1) < 1e-13:
            self.U = np.eye(d)
        elif np.linalg.norm(self.u1 + e1) < 1e-13:
            self.U = np.eye(d)
            self.U[0, 0] = -1
        else:
            v = (e1 - self.u1) / np.linalg.norm(e1 - self.u1)
            self.U = np.eye(d) - 2 * np.outer(v, v)
        assert np.allclose(self.U @ self.U.T, np.eye(d), atol=1e-11)
        assert np.allclose(self.U[:, 0], self.u1, atol=1e-11)

    @staticmethod
    def brownian_factor(n_steps: int, T: float, construction: str = 'cholesky'):
        if n_steps < 1 or T <= 0:
            raise ValueError('n_steps and T must be positive')
        t = np.linspace(T / n_steps, T, n_steps)
        C = np.minimum.outer(t, t)
        if construction in ('cholesky', 'standard'):
            return np.linalg.cholesky(C)
        if construction == 'pca':
            eigvals, eigvecs = np.linalg.eigh(C)
            order = np.argsort(eigvals)[::-1]
            V = eigvecs[:, order]
            for j in range(n_steps):
                if V[np.argmax(abs(V[:, j])), j] < 0:
                    V[:, j] *= -1
            return V * np.sqrt(eigvals[order])[None, :]
        raise ValueError('construction must be cholesky or pca')

    def brownian_levels(self, Z: np.ndarray):
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2 or Z.shape[1] != len(self.q):
            raise ValueError('Z must have shape (N,d)')
        return (Z @ self.U.T) @ self.A0.T

    def gaussian_levels(self, Z):
        return self.brownian_levels(Z)

    def stock_paths(self, Z, S0, r, sigma, T):
        d = len(self.q)
        t = np.linspace(T / d, T, d)
        W = self.brownian_levels(Z)
        return S0 * np.exp((r - 0.5 * sigma*sigma) * t[None, :] + sigma * W)

    def alignment_error(self, Z):
        W = self.brownian_levels(Z)
        return float(np.max(np.abs(W @ self.q - self.D * np.asarray(Z)[:, 0])))


def payoff_indicator_direction(option_kind: str, n_steps: int):
    """Direction in BROWNIAN LEVEL space for exactly alignable boundaries.

    Terminal digital uses W_T; geometric-average options use mean(W_t).
    For arithmetic Asian digitals or running-minimum barriers this returns
    no direction because a single exact linear Gaussian boundary is absent.
    """
    if option_kind in ('digital', 'digital_put', 'euro_digital'):
        q = np.zeros(n_steps); q[-1] = 1
        return q
    if option_kind in ('geo_digital', 'geo_digital_put', 'geo_knockout', 'geo_knockout_reverse'):
        return np.ones(n_steps) / n_steps
    raise NotImplementedError(f'No exact one-direction OT for {option_kind}')
