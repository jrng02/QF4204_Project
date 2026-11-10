"""
Sampling methods for generating standard normal samples.

Three orthogonal estimators:

  - 'mc'      : pseudo-random Monte Carlo
  - 'qmc'     : DETERMINISTIC Sobol' net (NO randomization, NO seed effect)
  - 'cprqmc'  : Cranley-Patterson randomized Sobol' (weakest RQMC)
  - 'rqmc'    : Owen-scrambled Sobol' (standard RQMC; Owen 1995)

"""

import numpy as np
from scipy.stats import qmc, norm
from typing import Optional


# Universal clipping bounds for the inverse-CDF transform
_EPS = 1e-10


class Sampler:
    """Sampling methods for generating standard normal samples."""

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    @staticmethod
    def mc(n_simulations: int, n_steps: int,
           seed: Optional[int] = None) -> np.ndarray:
        """
        Standard Monte Carlo: pseudo-random standard normal samples.

        Uses a local numpy Generator; does NOT touch global RNG state.
        """
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_simulations, n_steps))

    # ------------------------------------------------------------------
    # Deterministic QMC
    # ------------------------------------------------------------------

    @staticmethod
    def qmc(n_simulations: int, n_steps: int,
            seed: Optional[int] = None) -> np.ndarray:
        """
        DETERMINISTIC Sobol' QMC.
        """
        sobol = qmc.Sobol(d=n_steps, scramble=False, seed=None)
        U = sobol.random(n_simulations)
        U = np.clip(U, _EPS, 1.0 - _EPS)
        return norm.ppf(U)

    # ------------------------------------------------------------------
    # Randomized QMC
    # ------------------------------------------------------------------

    @staticmethod
    def cprqmc(n_simulations: int, n_steps: int,
               seed: Optional[int] = None) -> np.ndarray:
        """
        Cranley-Patterson randomized Sobol'.
        """
        sobol = qmc.Sobol(d=n_steps, scramble=False, seed=None)
        U = sobol.random(n_simulations)
        rng = np.random.default_rng(seed)
        shift = rng.random(n_steps)
        U = (U + shift) % 1.0
        U = np.clip(U, _EPS, 1.0 - _EPS)
        return norm.ppf(U)

    @staticmethod
    def rqmc(n_simulations: int, n_steps: int,
             seed: Optional[int] = None) -> np.ndarray:
        """
        Owen-scrambled Sobol' RQMC.
        """
        sobol = qmc.Sobol(d=n_steps, scramble=True, seed=seed)
        U = sobol.random(n_simulations)
        U = np.clip(U, _EPS, 1.0 - _EPS)
        return norm.ppf(U)

    # ------------------------------------------------------------------
    # Unified interface
    # ------------------------------------------------------------------

    @staticmethod
    def get_samples(method: str, n_simulations: int, n_steps: int,
                    seed: Optional[int] = None) -> np.ndarray:
        if method == 'mc':
            return Sampler.mc(n_simulations, n_steps, seed)
        elif method == 'qmc':
            return Sampler.qmc(n_simulations, n_steps, seed)
        elif method == 'cprqmc':
            return Sampler.cprqmc(n_simulations, n_steps, seed)
        elif method == 'rqmc':
            return Sampler.rqmc(n_simulations, n_steps, seed)
        else:
            raise ValueError(f"Unknown sampling method: {method}")


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    # Shape and finiteness
    for m in ('mc', 'qmc', 'cprqmc', 'rqmc'):
        Z = Sampler.get_samples(m, 1024, 16, seed=42)
        assert Z.shape == (1024, 16), f"{m}: wrong shape"
        assert Z.dtype == np.float64, f"{m}: wrong dtype"
        assert np.all(np.isfinite(Z)), f"{m}: non-finite entries"
    print("Shape / finiteness: OK")

    # Reproducibility
    for m in ('mc', 'qmc', 'cprqmc', 'rqmc'):
        Z1 = Sampler.get_samples(m, 100, 4, seed=7)
        Z2 = Sampler.get_samples(m, 100, 4, seed=7)
        assert np.array_equal(Z1, Z2), f"{m}: not reproducible with same seed"
    print("Reproducibility with same seed: OK")

    # QMC must be seed-INDEPENDENT (deterministic)
    Za = Sampler.get_samples('qmc', 100, 4, seed=1)
    Zb = Sampler.get_samples('qmc', 100, 4, seed=999)
    Zc = Sampler.get_samples('qmc', 100, 4, seed=None)
    assert np.array_equal(Za, Zb), "qmc: changed with different seed"
    assert np.array_equal(Za, Zc), "qmc: changed with seed=None"
    print("QMC determinism (seed-independent): OK")

    # RQMC must be seed-DEPENDENT
    Za = Sampler.get_samples('cprqmc', 100, 4, seed=1)
    Zb = Sampler.get_samples('cprqmc', 100, 4, seed=2)
    assert not np.allclose(Za, Zb), "cprqmc: did not change with seed"
    Za = Sampler.get_samples('rqmc', 100, 4, seed=1)
    Zb = Sampler.get_samples('rqmc', 100, 4, seed=2)
    assert not np.allclose(Za, Zb), "rqmc: did not change with seed"
    print("RQMC seed-dependence: OK")

    # No global-state contamination

    a = Sampler.mc(50, 4, seed=1)
    np.random.seed(12345)      # deliberately pollute global state
    b = Sampler.mc(50, 4, seed=1)
    assert np.array_equal(a, b), "mc: global RNG pollution detected"
    print("No global-state contamination: OK")

    # Distributional sanity
    for m in ('mc', 'cprqmc', 'rqmc'):
        Z = Sampler.get_samples(m, 2**17, 4, seed=11)
        assert abs(Z.mean()) < 0.02, f"{m}: mean drifted"
        assert abs(Z.var() - 1.0) < 0.02, f"{m}: variance drifted"
    print("Distributional sanity: OK")

    print("\nAll samplers self-tests passed.")