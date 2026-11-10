"""
conditional_qmc.py

Conditional Quasi-Monte Carlo (CQMC) smoothing for discontinuous payoffs.

Implements the two-step framework of Xiao & Wang (2018):
  "Conditional quasi-Monte Carlo methods and dimension reduction for
   option pricing and hedging with discontinuous functions",
   Journal of Computational and Applied Mathematics, 337:193-209.

Step 1 (this file): Conditional QMC smoothing
  - Decompose the Brownian path so that W(t_1) is generated separately
    from the remaining coordinates W(t_2),...,W(t_d).
  - Variable separation:
        g(x) = h(x) * 1{ psi_d(z) < x_1 < psi_u(z) }
    where x_1 = W(t_1) and z = (x_2, ..., x_d).
  - The conditional expectation G(z) = E[g | z] is computed analytically.
  - The result G(z) is SMOOTH (continuous), removing the discontinuity
    that degrades QMC performance.

Step 2 (separate file): Gradient-based PCA (GPCA) on the smoothed function.

"""

import numpy as np
from scipy.stats import norm
from typing import Optional, Tuple, Callable
import warnings


class ConditionalQMC:
    """
    Conditional QMC smoothing for discontinuous payoffs.

    Variable-separated path generation:
        W(t_1) = sqrt(t_1) * Z_1
        (W(t_2), ..., W(t_d)) = W(t_1) + A @ Z_{2:d}
    where A is the Cholesky factor of the conditional covariance matrix.
    """

    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float,
                 n_steps: int, barrier: Optional[float] = None,
                 barrier_type: str = 'down-out',
                 digital_payoff: float = 1.0):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_steps = n_steps
        self.barrier = barrier
        self.barrier_type = barrier_type
        self.digital_payoff = digital_payoff

        self.dt = T / n_steps
        self.times = np.linspace(self.dt, T, n_steps)
        self.drift = (r - 0.5 * sigma**2) * self.dt
        self.diffusion = sigma * np.sqrt(self.dt)

        self._build_path_decomposition()

    # ------------------------------------------------------------------
    # Path decomposition
    # ------------------------------------------------------------------

    def _build_path_decomposition(self):
        """
        Build the decomposition:
            W(t_1) = sqrt(t_1) * Z_1
            W(t_j) = W(t_1) + A[j-1, :] @ Z_{2:d}   for j = 2, ..., n_steps
        """
        d = self.n_steps
        t = self.times

        n_cond = d - 1
        Sigma = np.zeros((n_cond, n_cond))
        for i in range(n_cond):
            for j in range(n_cond):
                Sigma[i, j] = min(t[i + 1], t[j + 1]) - t[0]

        Sigma += 1e-14 * np.eye(n_cond)
        self.A = np.linalg.cholesky(Sigma)  # (d-1, d-1)
        self.t1 = t[0]

    # ------------------------------------------------------------------
    # Path generation from separated coordinates
    # ------------------------------------------------------------------

    def generate_paths_from_separated(self, Z1: np.ndarray,
                                       Z_rest: np.ndarray) -> np.ndarray:
        """Generate stock paths from separated coordinates."""
        n_sims = Z1.shape[0]
        d = self.n_steps

        W1 = np.sqrt(self.t1) * Z1
        dW = Z_rest @ self.A.T

        W = np.zeros((n_sims, d))
        W[:, 0] = W1
        W[:, 1:] = W1[:, None] + dW

        log_paths = (np.log(self.S0)
                     + (self.r - 0.5 * self.sigma**2) * self.times
                     + self.sigma * W)
        return np.exp(log_paths)

    # ------------------------------------------------------------------
    # Conditional expectation: digital
    # ------------------------------------------------------------------

    def conditional_expectation_digital(self, Z_rest: np.ndarray,
                                         K: Optional[float] = None,
                                         payoff: Optional[float] = None
                                         ) -> np.ndarray:
        """
        Compute E[ payoff * 1{S_T >= K} | Z_rest ] analytically.

        Given Z_rest, S_T = S_0 exp(mu_cond + a Z_1) where
            mu_cond = (r - sigma^2/2) T + sigma (W(T) - W(t_1))
            a       = sigma sqrt(t_1)
        The condition S_T >= K becomes Z_1 >= threshold.
        """
        if K is None:
            K = self.K
        if payoff is None:
            payoff = self.digital_payoff

        # W(T) - W(t_1) from last row of A
        dW_last = Z_rest @ self.A[-1, :]
        mu_cond = (self.r - 0.5 * self.sigma**2) * self.T + self.sigma * dW_last

        a = self.sigma * np.sqrt(self.t1)
        threshold = (np.log(K / self.S0) - mu_cond) / a

        prob = 1.0 - norm.cdf(threshold)
        return payoff * prob  # NOT discounted here; caller discounts

    # ------------------------------------------------------------------
    # Conditional expectation: down-and-out barrier call
    # ------------------------------------------------------------------

    def conditional_expectation_barrier_call(self, Z_rest: np.ndarray,
                                              K: Optional[float] = None,
                                              barrier: Optional[float] = None,
                                              barrier_type: Optional[str] = None
                                              ) -> np.ndarray:
        """
        Compute E[ e^{-rT} (S_T - K)^+ * 1{ min_{1<=j<=d} S_j > L } | Z_rest ]
        analytically for a DOWN-AND-OUT barrier call.

        The path factorizes as
            S_j = S_j_tilde * C,    C = exp( a Z_1 ),  a = sigma sqrt(t_1)
        where S_j_tilde is the path computed from Z_rest alone (without
        the Z_1 factor).  The barrier and strike conditions become lower
        bounds on C:

            S_T >= K        <=>  C >= K / S_T_tilde
            S_1 > L         <=>  C >  L / S_1_tilde
            min_{j>=2} S_j > L  <=>  C >  L / min_{j>=2} S_j_tilde

        So C_min = max( K / S_T_tilde,
                        L / S_1_tilde,
                        L / min_{j>=2} S_j_tilde )

        The conditional expectation is computed using the truncated
        lognormal moment identity:
            E[ e^{aZ} 1{Z >= b} ] = e^{a^2/2} (1 - Phi(b - a))
            P( Z >= b )           = 1 - Phi(b)
        """
        if K is None:
            K = self.K
        if barrier is None:
            barrier = self.barrier
        if barrier_type is None:
            barrier_type = self.barrier_type

        if barrier is None:
            raise ValueError("Barrier level required for barrier option.")

        if barrier_type != 'down-out':
            raise NotImplementedError(
                "Conditional barrier smoothing supports only down-out calls."
            )

        # Path from Z_rest alone (excluding the Z_1 factor)
        dW = Z_rest @ self.A.T                    # (n_sims, d-1)
        log_paths_tilde = (np.log(self.S0)
                           + (self.r - 0.5 * self.sigma**2) * self.times[1:]
                           + self.sigma * dW)
        S_tilde = np.exp(log_paths_tilde)          # (n_sims, d-1)

        S_T_tilde = S_tilde[:, -1]
        min_S_tilde = np.min(S_tilde, axis=1)

        # t_1 asset price (without the Z_1 factor):
        #   S_1_tilde = S_0 exp( (r - sigma^2/2) t_1 )
        S_1_tilde = self.S0 * np.exp(
            (self.r - 0.5 * self.sigma**2) * self.t1
        )

        # Effective lower bound on C = exp(a Z_1)
        # IMPORTANT: include the t_1 barrier check!
        C_min = np.maximum(
            np.maximum(K / S_T_tilde, barrier / min_S_tilde),
            barrier / S_1_tilde,
        )

        a = self.sigma * np.sqrt(self.t1)
        threshold = np.log(C_min) / a

        # E[ (C S_T_tilde - K) 1{C >= C_min} ]
        #   = S_T_tilde * E[ C 1{Z_1 >= threshold} ]
        #     - K * P(Z_1 >= threshold)
        term1 = S_T_tilde * np.exp(0.5 * a**2) * (1.0 - norm.cdf(threshold - a))
        term2 = K * (1.0 - norm.cdf(threshold))

        discounted = np.exp(-self.r * self.T)
        return discounted * np.maximum(term1 - term2, 0.0)

    # ------------------------------------------------------------------
    # Unified dispatcher
    # ------------------------------------------------------------------

    def conditional_expectation(self, Z_rest: np.ndarray,
                                 option_type: str,
                                 **kwargs) -> np.ndarray:
        """
        Dispatch to the correct conditional expectation.

        option_type : 'digital' or 'barrier_down_out'
        """
        if option_type == 'digital':
            return self.conditional_expectation_digital(Z_rest, **kwargs)
        elif option_type == 'barrier_down_out':
            return self.conditional_expectation_barrier_call(Z_rest, **kwargs)
        else:
            raise ValueError(f"Unknown option_type: {option_type}")

    # ------------------------------------------------------------------
    # Sampling interface
    # ------------------------------------------------------------------

    def sample_conditional(self, n_simulations: int,
                           sampler: Callable,
                           option_type: str,
                           seed: Optional[int] = None,
                           **kwargs) -> np.ndarray:
        """
        Draw samples of the smoothed integrand G(Z_rest).

        Returns
        -------
        smoothed_payoffs : np.ndarray, shape (n_simulations,)
            Already DISCOUNTED (both digital and barrier cases).
        """
        Z_rest = sampler(n_simulations, self.n_steps - 1, seed)
        smoothed = self.conditional_expectation(Z_rest, option_type, **kwargs)

        # Both estimators should return undiscounted payoffs from
        # conditional_expectation; the barrier case already includes
        # the discount factor, so normalize here.
        if option_type == 'digital':
            smoothed = np.exp(-self.r * self.T) * smoothed
        # For barrier, the discount is already applied inside.

        return smoothed


# ======================================================================
# Convenience wrapper
# ======================================================================

def price_with_conditional_qmc(
    S0: float, K: float, T: float, r: float, sigma: float,
    n_steps: int, n_simulations: int,
    option_type: str,
    sampler: Callable,
    seed: Optional[int] = None,
    barrier: Optional[float] = None,
    digital_payoff: float = 1.0,
    **kwargs
) -> Tuple[float, float]:
    """Return (price, std_error) using conditional QMC smoothing."""
    cqmc = ConditionalQMC(
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        n_steps=n_steps, barrier=barrier,
        digital_payoff=digital_payoff,
    )
    smoothed = cqmc.sample_conditional(
        n_simulations=n_simulations, sampler=sampler,
        option_type=option_type, seed=seed, **kwargs
    )
    price = float(np.mean(smoothed))
    se = float(np.std(smoothed, ddof=1) / np.sqrt(len(smoothed)))
    return price, se


# ======================================================================
# Self-test
# ======================================================================

if __name__ == "__main__":
    from scipy.stats import qmc
    from benchmarks import OptionPricingMC, down_and_out_call

    def rqmc_sampler(n, d, seed):
        s = qmc.Sobol(d=d, scramble=True, seed=seed)
        U = s.random(n)
        return norm.ppf(np.clip(U, 1e-10, 1 - 1e-10))

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    n_steps = 16
    n_sims = 2 ** 15   # power of 2 for Sobol'

    print("=" * 60)
    print("Conditional QMC self-test (v2)")
    print("=" * 60)

    # ---- Digital ----
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    bs_digital = np.exp(-r * T) * norm.cdf(d2)
    print(f"\n[Digital] BS reference:  {bs_digital:.6f}")

    price_d, se_d = price_with_conditional_qmc(
        S0, K, T, r, sigma, n_steps, n_sims,
        option_type='digital', sampler=rqmc_sampler, seed=42,
    )
    print(f"[Digital] CQMC:           {price_d:.6f} (±{se_d:.6f})")
    print(f"[Digital] Error:          {abs(price_d - bs_digital):.6f}")
    print(f"[Digital] Z-score:        {abs(price_d - bs_digital) / se_d:.2f}")

    # ---- Barrier ----
    barrier = 80.0
    ref_disc, ref_se = OptionPricingMC.discrete_barrier_reference(
        S0, K, T, r, sigma, barrier, 'down-out',
        n_steps=n_steps, n_ref=2_000_000, seed=999
    )
    ref_cont = down_and_out_call(S0, K, barrier, r, sigma, T)

    price_b, se_b = price_with_conditional_qmc(
        S0, K, T, r, sigma, n_steps, n_sims,
        option_type='barrier_down_out', sampler=rqmc_sampler,
        barrier=barrier, seed=42,
    )
    print(f"\n[Barrier] Discrete MC ref: {ref_disc:.6f} (±{ref_se:.6f})")
    print(f"[Barrier] Continuous Merton: {ref_cont:.6f}")
    print(f"[Barrier] CQMC:             {price_b:.6f} (±{se_b:.6f})")
    print(f"[Barrier] Error vs discrete:{abs(price_b - ref_disc):.6f}")
    print(f"[Barrier] Z-score:          {abs(price_b - ref_disc) / ref_se:.2f}")

    print("\nSelf-test complete.")
# ----------------------------------------------------------------------
# Extended exact one-coordinate conditional integration for GBM payoffs.
# This uses the independent first Brownian increment W(t1). Conditional
# stock paths factor as S_j = S_tilde_j * exp(a * Z1), for every j.
# ----------------------------------------------------------------------
class StructuredConditionalQMC(ConditionalQMC):
    """Analytic conditional estimators, with residual BB/PCA constructions.

    All returned values are DISCOUNTED option prices per simulated residual
    path; do not discount again in the caller. Monitoring uses t1,...,T.
    """
    def _residual_factor(self, construction='standard'):
        d = self.n_steps - 1
        if d <= 0:
            return np.zeros((0, 0))
        if construction == 'standard':
            return self.A
        t = np.arange(1, d+1) * self.dt
        C = np.minimum.outer(t, t)
        if construction == 'pca':
            vals, V = np.linalg.eigh(C)
            order = np.argsort(vals)[::-1]
            V = V[:, order]
            for j in range(d):
                if V[np.argmax(np.abs(V[:, j])),j] < 0:
                    V[:,j] *= -1
            return V * np.sqrt(vals[order])[None,:]
        if construction == 'bb':
            # General midpoint Brownian bridge, works even when d is not 2^k.
            A = np.zeros((d, d))
            A[-1,0] = np.sqrt(d*self.dt)
            k = 1
            intervals = [(0, d)]
            while intervals:
                next_intervals = []
                for left,right in intervals:
                    if right-left <= 1: continue
                    mid = (left+right)//2
                    tl,tm,tr = left*self.dt,mid*self.dt,right*self.dt
                    v = (tm-tl)*(tr-tm)/(tr-tl)
                    a = (tr-tm)/(tr-tl)
                    b = (tm-tl)/(tr-tl)
                    A[mid-1] = (a*A[left-1] if left else 0) + b*A[right-1]
                    A[mid-1,k] += np.sqrt(v)
                    k += 1
                    next_intervals.extend([(left,mid),(mid,right)])
                intervals = next_intervals
            if k != d: raise RuntimeError('Brownian bridge fill incomplete')
            return A
        raise ValueError(f'Unknown conditional construction: {construction}')

    def conditional_values(self, Z_rest, kind, construction='standard', K=None, H=None,
                           cash=None):
        from scipy.special import ndtr
        Z_rest = np.asarray(Z_rest)
        if Z_rest.ndim != 2 or Z_rest.shape[1] != self.n_steps-1:
            raise ValueError('Z_rest shape mismatch')
        if self.n_steps < 2:
            raise ValueError('Conditional dimension must be >= 2')
        K = self.K if K is None else K
        H = self.barrier if H is None else H
        cash = self.digital_payoff if cash is None else cash
        W_tail = Z_rest @ self._residual_factor(construction).T
        W = np.column_stack([np.zeros(len(Z_rest)), W_tail])
        tilde = self.S0 * np.exp((self.r-.5*self.sigma**2)*self.times[None,:]
                                 + self.sigma*W)
        st = tilde[:,-1]
        avg = tilde.mean(axis=1)
        geo = np.exp(np.log(tilde).mean(axis=1))
        a = self.sigma*np.sqrt(self.t1)
        if a <= 0: raise ValueError('sigma must be positive')
        # For C=exp(a*Z), P(C>=x)=Phi(-log(x)/a),
        # E[C 1(C>=x)]=exp(a^2/2) Phi(a-log(x)/a).
        def tail(x, moment=False):
            x = np.maximum(x, 1e-300)
            arg = np.log(x)/a
            return np.exp(.5*a*a)*ndtr(a-arg) if moment else ndtr(-arg)
        if kind == 'digital':
            out = cash*tail(K/st)
        elif kind == 'digital_put':
            out = cash*(1.-tail(K/st))
        elif kind == 'geo_digital':
            out = cash*tail(K/geo)
        elif kind == 'geo_digital_put':
            out = cash*(1.-tail(K/geo))
        elif kind == 'arith_digital':
            out = cash*tail(K/avg)
        elif kind == 'arith_digital_put':
            out = cash*(1.-tail(K/avg))
        elif kind in ('barrier','barrier_up_out_call','geo_knockout','geo_knockout_reverse','arith_call'):
            base = st if kind in ('barrier','barrier_up_out_call') else avg
            lo = K/base
            if kind == 'barrier':
                if H is None: raise ValueError('H is required')
                lo = np.maximum(lo, H/tilde.min(axis=1))
                out = base*tail(lo, True)-K*tail(lo)
            elif kind == 'barrier_up_out_call':
                if H is None: raise ValueError('H is required')
                hi = H/tilde.max(axis=1)
                out = np.where(hi>lo,base*(tail(lo, True)-tail(hi, True))
                               -K*(tail(lo)-tail(hi)),0.)
            elif kind == 'geo_knockout':
                if H is None: raise ValueError('H is required')
                hi = H/geo
                out = np.where(hi>lo, base*(tail(lo, True)-tail(hi, True))
                               -K*(tail(lo)-tail(hi)), 0.)
            elif kind == 'geo_knockout_reverse':
                if H is None: raise ValueError('H is required')
                lo = np.maximum(lo,H/geo)
                out = base*tail(lo,True)-K*tail(lo)
            else:
                out = base*tail(lo, True)-K*tail(lo)
        else:
            raise NotImplementedError(f'No conditional formula for {kind}')
        return np.exp(-self.r*self.T)*np.maximum(out, 0.)
