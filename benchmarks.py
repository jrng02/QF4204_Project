"""
Benchmark option pricing models and Monte Carlo option pricing class.

Closed-form references:
  - Vanilla European: Black-Scholes
  - Digital (cash-or-nothing): Black-Scholes
  - Geometric Asian (fixed/floating strike): Kemna-Vorst
  - Barrier (continuous): Merton (1973)

Discrete reference prices:
  - Barrier: high-N MC using the SAME discretization as the estimator.
  - Digital Asian / Barrier Asian: high-N MC (no closed form).

References are cached in a module-level dict to avoid recomputing the
2M-path MC every time OptionPricingMC is instantiated.
"""

import numpy as np
from scipy.stats import norm, qmc
from typing import Tuple, Optional, Dict


# ==========================================
# Module-level reference price cache
# ==========================================

_REFERENCE_CACHE: Dict[tuple, Tuple[float, float]] = {}


# ==========================================
# CLOSED-FORM PRICING FORMULAS
# ==========================================

def bs_vanilla(S0, X, r, T, sigma, option_type='call', q=0):
    """Black-Scholes price for vanilla European option."""
    d1 = (np.log(S0 / X) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        V0 = S0 * np.exp(-q * T) * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    else:
        V0 = X * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    return V0


def BS_digital(S0, K, X, r, T, sigma, option_type='call'):
    """
    Black-Scholes price for a cash-or-nothing digital option.

    Payoff: X * 1{S_T >= K}  (call)
            X * 1{S_T <= K}  (put)
    """
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        V0 = X * np.exp(-r * T) * norm.cdf(d2)
    else:
        V0 = X * np.exp(-r * T) * norm.cdf(-d2)
    return V0


def _geo_log_moments(S0, r, sigma, T, N, sampling):
    if sampling == 'continuous':
        mean_t = T / 2
        brownian_variance = T / 3
        cov_terminal_average = T / 2
    elif sampling == 'discrete':
        if N < 1:
            raise ValueError('N must be positive')
        mean_t = T * (N + 1) / (2 * N)
        brownian_variance = T * (N + 1) * (2 * N + 1) / (6 * N * N)
        cov_terminal_average = mean_t
    else:
        raise ValueError('sampling must be continuous or discrete')
    mean_log_g = np.log(S0) + (r - 0.5 * sigma**2) * mean_t
    var_log_g = sigma**2 * brownian_variance
    mean_log_st = np.log(S0) + (r - 0.5 * sigma**2) * T
    var_log_st = sigma**2 * T
    cov = sigma**2 * cov_terminal_average
    return mean_log_g, var_log_g, mean_log_st, var_log_st, cov


def KV_geo_asian_fixedstrike(S0, K, r, sigma, T, N, option_type='call',
                              sampling='continuous'):
    """Exact lognormal geometric-average fixed-strike price under GBM."""
    mu, var, *_ = _geo_log_moments(S0, r, sigma, T, N, sampling)
    if var <= 0:
        return np.exp(-r*T) * (max(np.exp(mu)-K, 0) if option_type == 'call'
                               else max(K-np.exp(mu), 0))
    sd = np.sqrt(var)
    d2 = (mu - np.log(K)) / sd
    d1 = d2 + sd
    forward = np.exp(mu + var / 2)
    if option_type == 'call':
        return np.exp(-r*T) * (forward * norm.cdf(d1) - K * norm.cdf(d2))
    if option_type == 'put':
        return np.exp(-r*T) * (K * norm.cdf(-d2) - forward * norm.cdf(-d1))
    raise ValueError('option_type must be call or put')


def KV_geo_asian_floatingstrike(S0, r, sigma, T, N, option_type='call',
                                 sampling='continuous'):
    """Exact joint-lognormal price for max(S_T-G,0) or max(G-S_T,0)."""
    mg, vg, ms, vs, cov = _geo_log_moments(S0, r, sigma, T, N, sampling)
    md = ms - mg
    vd = max(vs + vg - 2 * cov, 0.0)
    eg = np.exp(mg + vg / 2)
    es = np.exp(ms + vs / 2)
    if vd <= 1e-15:
        return np.exp(-r*T) * (max(es-eg, 0) if option_type == 'call'
                               else max(eg-es, 0))
    sd = np.sqrt(vd)
    ds = (md + vs - cov) / sd
    dg = (md + cov - vg) / sd
    if option_type == 'call':
        return np.exp(-r*T) * (es * norm.cdf(ds) - eg * norm.cdf(dg))
    if option_type == 'put':
        return np.exp(-r*T) * (eg * norm.cdf(-dg) - es * norm.cdf(-ds))
    raise ValueError('option_type must be call or put')


def _d_plus(x, K, r, sigma, tau):
    return (np.log(x / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))


def _d_minus(x, K, r, sigma, tau):
    return (np.log(x / K) + (r - 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))


# ==========================================
# CONTINUOUS BARRIER (MERTON 1973)
# ==========================================

def down_and_out_call(S0, K, B, r, sigma, T):
    """Down-and-Out Call, continuous monitoring (Merton 1973)."""
    if B >= S0:
        raise ValueError("Barrier must be below S0 for a down-and-out call.")

    dp = _d_plus(S0, K, r, sigma, T)
    dm = _d_minus(S0, K, r, sigma, T)
    dp_refl = _d_plus(B**2 / S0, K, r, sigma, T)
    dm_refl = _d_minus(B**2 / S0, K, r, sigma, T)

    vanilla_part = S0 * norm.cdf(dp) - K * np.exp(-r * T) * norm.cdf(dm)
    reflected_part = S0 * (B / S0)**(1 + 2 * r / sigma**2) * norm.cdf(dp_refl) \
        - K * np.exp(-r * T) * (B / S0)**(2 * r / sigma**2 - 1) * norm.cdf(dm_refl)
    return vanilla_part - reflected_part


def down_and_out_put(S0, K, H, r, sigma, T):
    """Down-and-Out Put, continuous monitoring."""
    if S0 <= H:
        return 0.0

    d1 = _d_plus(S0, K, r, sigma, T)
    d2 = _d_minus(S0, K, r, sigma, T)
    d1_refl = _d_plus(H**2 / S0, K, r, sigma, T)
    d2_refl = _d_minus(H**2 / S0, K, r, sigma, T)

    lam = (2 * r) / sigma**2

    vanilla_part = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    reflected_part = (K * np.exp(-r * T) * (H / S0)**(lam - 1) * norm.cdf(-d2_refl) -
                      S0 * (H / S0)**(1 + lam) * norm.cdf(-d1_refl))

    return max(vanilla_part - reflected_part, 0.0)


def down_and_in_call(S0, K, H, r, sigma, T):
    """Down-and-In Call, continuous monitoring."""
    vanilla = bs_vanilla(S0, K, r, T, sigma, 'call')
    do = down_and_out_call(S0, K, H, r, sigma, T)
    return max(vanilla - do, 0.0)


def down_and_in_put(S0, K, H, r, sigma, T):
    """Down-and-In Put, continuous monitoring."""
    vanilla = bs_vanilla(S0, K, r, T, sigma, 'put')
    do = down_and_out_put(S0, K, H, r, sigma, T)
    return max(vanilla - do, 0.0)


def up_and_out_call(S0, K, H, r, sigma, T):
    """Up-and-Out Call, continuous monitoring."""
    if S0 >= H:
        return 0.0

    d1 = _d_plus(S0, K, r, sigma, T)
    d2 = _d_minus(S0, K, r, sigma, T)
    d1_refl = _d_plus(H**2 / S0, K, r, sigma, T)
    d2_refl = _d_minus(H**2 / S0, K, r, sigma, T)

    lam = (2 * r) / sigma**2

    vanilla_part = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    reflected_part = (S0 * (H / S0)**(1 + lam) * norm.cdf(d1_refl) -
                      K * np.exp(-r * T) * (H / S0)**(lam - 1) * norm.cdf(d2_refl))

    return max(vanilla_part - reflected_part, 0.0)


def up_and_out_put(S0, K, H, r, sigma, T):
    """Up-and-Out Put, continuous monitoring."""
    if S0 >= H:
        return 0.0

    d1 = _d_plus(S0, K, r, sigma, T)
    d2 = _d_minus(S0, K, r, sigma, T)
    d1_refl = _d_plus(H**2 / S0, K, r, sigma, T)
    d2_refl = _d_minus(H**2 / S0, K, r, sigma, T)

    lam = (2 * r) / sigma**2

    vanilla_part = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    reflected_part = (K * np.exp(-r * T) * (H / S0)**(lam - 1) * norm.cdf(-d2_refl) -
                      S0 * (H / S0)**(1 + lam) * norm.cdf(-d1_refl))

    return max(vanilla_part - reflected_part, 0.0)


def up_and_in_call(S0, K, H, r, sigma, T):
    """Up-and-In Call, continuous monitoring."""
    vanilla = bs_vanilla(S0, K, r, T, sigma, 'call')
    uo = up_and_out_call(S0, K, H, r, sigma, T)
    return max(vanilla - uo, 0.0)


def up_and_in_put(S0, K, H, r, sigma, T):
    """Up-and-In Put, continuous monitoring."""
    vanilla = bs_vanilla(S0, K, r, T, sigma, 'put')
    uo = up_and_out_put(S0, K, H, r, sigma, T)
    return max(vanilla - uo, 0.0)


def barrier_option(S0, K, H, r, sigma, T, barrier_type, option_type='call'):
    """Unified CONTINUOUS barrier option pricing (Merton)."""
    if barrier_type == 'down-out':
        return down_and_out_call(S0, K, H, r, sigma, T) if option_type == 'call' \
            else down_and_out_put(S0, K, H, r, sigma, T)
    elif barrier_type == 'down-in':
        return down_and_in_call(S0, K, H, r, sigma, T) if option_type == 'call' \
            else down_and_in_put(S0, K, H, r, sigma, T)
    elif barrier_type == 'up-out':
        return up_and_out_call(S0, K, H, r, sigma, T) if option_type == 'call' \
            else up_and_out_put(S0, K, H, r, sigma, T)
    elif barrier_type == 'up-in':
        return up_and_in_call(S0, K, H, r, sigma, T) if option_type == 'call' \
            else up_and_in_put(S0, K, H, r, sigma, T)
    else:
        raise ValueError("barrier_type must be 'down-out', 'down-in', 'up-out', or 'up-in'")


def barrier_option_bgk(S0, K, H, r, sigma, T, barrier_type,
                       option_type='call', n_steps=252):
    """
    BGK (Broadie-Glasserman-Kou 1997) continuity-corrected DISCRETE
    barrier price. First-order accurate; error is O(dt).

    Use this as a sanity check, not as the primary reference. For
    rigorous comparison, use OptionPricingMC.discrete_barrier_reference.
    """
    dt = T / n_steps
    beta = 0.5826
    shift = np.exp(beta * sigma * np.sqrt(dt))
    if 'down' in barrier_type:
        H_adj = H / shift
    else:
        H_adj = H * shift
    return barrier_option(S0, K, H_adj, r, sigma, T, barrier_type, option_type)


# ==========================================
# MONTE CARLO OPTION PRICING CLASS
# ==========================================

class OptionPricingMC:
    """
    Monte Carlo and Quasi-Monte Carlo option pricing for various option types.

    Supported option types
    ----------------------
    - 'call' / 'put'            : vanilla European
    - 'digital'                 : cash-or-nothing
    - 'asian'                   : Asian (geometric or arithmetic)
    - 'asian_floating'          : floating-strike geometric Asian
    - 'barrier'                 : discrete barrier
    - 'digital_asian'           : digital on the Asian average
    - 'barrier_asian'           : barrier on the geometric average
    """

    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float,
                 n_simulations: int = 100000, n_steps: int = 252,
                 option_type: str = 'call',
                 barrier: Optional[float] = None,
                 barrier_type: str = 'down-out',
                 digital_payoff: float = 1.0,
                 asian_averaging: str = 'geometric',
                 true_price_override: Optional[float] = None):

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.option_type = option_type
        self.barrier = barrier
        self.barrier_type = barrier_type
        self.digital_payoff = digital_payoff
        self.asian_averaging = asian_averaging

        self.dt = T / n_steps
        self.drift = (r - 0.5 * sigma**2) * self.dt
        self.diffusion = sigma * np.sqrt(self.dt)

        if true_price_override is not None:
            self.closed_form_price = true_price_override
        else:
            self.closed_form_price = self.get_closed_form_price()

    # ------------------------------------------------------------------
    # Reference price dispatch
    # ------------------------------------------------------------------

    def get_closed_form_price(self) -> float:
        """Return the reference price for this option."""
        if self.option_type in ['call', 'put']:
            return bs_vanilla(self.S0, self.K, self.r, self.T, self.sigma,
                              self.option_type)

        elif self.option_type == 'digital':
            return BS_digital(self.S0, self.K, self.digital_payoff,
                              self.r, self.T, self.sigma, 'call')

        elif self.option_type == 'asian':
            if self.asian_averaging == 'geometric':
                return KV_geo_asian_fixedstrike(
                    self.S0, self.K, self.r, self.sigma, self.T,
                    self.n_steps, 'call', sampling='discrete'
                )
            else:
                return self.arithmetic_asian_reference()[0]

        elif self.option_type == 'asian_floating':
            return KV_geo_asian_floatingstrike(
                self.S0, self.r, self.sigma, self.T, self.n_steps,
                'call', sampling='discrete'
            )

        elif self.option_type == 'barrier':
            if self.barrier is None:
                return bs_vanilla(self.S0, self.K, self.r, self.T,
                                  self.sigma, 'call')
            return self._cached_barrier_reference(
                self.S0, self.K, self.T, self.r, self.sigma,
                self.barrier, self.barrier_type, self.n_steps
            )[0]

        elif self.option_type == 'digital_asian':
            return self._cached_digital_asian_reference(
                self.S0, self.K, self.T, self.r, self.sigma,
                self.n_steps, self.asian_averaging, self.digital_payoff
            )[0]

        elif self.option_type == 'barrier_asian':
            if self.barrier is None:
                raise ValueError("Barrier level required for barrier_asian.")
            return self._cached_barrier_asian_reference(
                self.S0, self.K, self.T, self.r, self.sigma,
                self.barrier, self.barrier_type, self.n_steps,
                self.asian_averaging
            )[0]

        else:
            raise ValueError(f"Unknown option type: {self.option_type}")

    # ------------------------------------------------------------------
    # Cached references
    # ------------------------------------------------------------------

    @staticmethod
    def _cached_barrier_reference(S0, K, T, r, sigma, H, btype, n_steps,
                                  n_ref=2_000_000, seed=999):
        key = ('barrier', S0, K, T, r, sigma, H, btype, n_steps, n_ref)
        if key not in _REFERENCE_CACHE:
            _REFERENCE_CACHE[key] = OptionPricingMC.discrete_barrier_reference(
                S0, K, T, r, sigma, H, btype, n_steps, n_ref, seed)
        return _REFERENCE_CACHE[key]

    @staticmethod
    def _cached_digital_asian_reference(S0, K, T, r, sigma, n_steps,
                                        averaging, payoff,
                                        n_ref=2_000_000, seed=999):
        key = ('digital_asian', S0, K, T, r, sigma, n_steps, averaging, payoff, n_ref)
        if key not in _REFERENCE_CACHE:
            _REFERENCE_CACHE[key] = OptionPricingMC.digital_asian_reference(
                S0, K, T, r, sigma, n_steps, averaging, payoff, n_ref, seed)
        return _REFERENCE_CACHE[key]

    @staticmethod
    def _cached_barrier_asian_reference(S0, K, T, r, sigma, H, btype,
                                        n_steps, averaging,
                                        n_ref=2_000_000, seed=999):
        key = ('barrier_asian', S0, K, T, r, sigma, H, btype, n_steps, averaging, n_ref)
        if key not in _REFERENCE_CACHE:
            _REFERENCE_CACHE[key] = OptionPricingMC.barrier_asian_reference(
                S0, K, T, r, sigma, H, btype, n_steps, averaging, n_ref, seed)
        return _REFERENCE_CACHE[key]

    # ------------------------------------------------------------------
    # High-N MC reference prices
    # ------------------------------------------------------------------

    @staticmethod
    def discrete_barrier_reference(S0, K, T, r, sigma, H, barrier_type,
                                    n_steps, n_ref=2_000_000, seed=999):
        """
        High-N MC reference for a DISCRETE barrier option.
        Uses the same n_steps as the estimator being tested.
        """
        dt = T / n_steps
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_ref, n_steps))
        log_paths = np.cumsum(drift + diffusion * Z, axis=1)
        stock_paths = S0 * np.exp(log_paths)
        final_prices = stock_paths[:, -1]

        if barrier_type == 'down-out':
            min_prices = np.min(stock_paths, axis=1)
            payoffs = np.where(min_prices > H,
                               np.maximum(final_prices - K, 0), 0.0)
        elif barrier_type == 'up-out':
            max_prices = np.max(stock_paths, axis=1)
            payoffs = np.where(max_prices < H,
                               np.maximum(final_prices - K, 0), 0.0)
        elif barrier_type == 'down-in':
            min_prices = np.min(stock_paths, axis=1)
            payoffs = np.where(min_prices <= H,
                               np.maximum(final_prices - K, 0), 0.0)
        elif barrier_type == 'up-in':
            max_prices = np.max(stock_paths, axis=1)
            payoffs = np.where(max_prices >= H,
                               np.maximum(final_prices - K, 0), 0.0)
        else:
            raise ValueError(f"Unknown barrier_type: {barrier_type}")

        discounted = np.exp(-r * T) * payoffs
        price = float(np.mean(discounted))
        se = float(np.std(discounted, ddof=1) / np.sqrt(n_ref))
        return price, se

    def arithmetic_asian_reference(self, n_ref=2_000_000, seed=999):
        """
        High-N MC reference for an arithmetic Asian call.
        """
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_ref, self.n_steps))
        log_paths = np.cumsum(self.drift + self.diffusion * Z, axis=1)
        stock_paths = self.S0 * np.exp(log_paths)
        arith_avg = np.mean(stock_paths, axis=1)
        payoffs = np.maximum(arith_avg - self.K, 0)
        discounted = np.exp(-self.r * self.T) * payoffs
        price = float(np.mean(discounted))
        se = float(np.std(discounted, ddof=1) / np.sqrt(n_ref))
        return price, se

    @staticmethod
    def digital_asian_reference(S0, K, T, r, sigma, n_steps,
                                 averaging='geometric', payoff=100.0,
                                 n_ref=2_000_000, seed=999):
        """
        High-N MC reference for a digital-on-the-Asian-average option.

        Payoff: payoff * 1{ A >= K }  where A is the (geometric or
        arithmetic) average of the price path.
        """
        dt = T / n_steps
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_ref, n_steps))
        log_paths = np.cumsum(drift + diffusion * Z, axis=1)
        paths = S0 * np.exp(log_paths)

        if averaging == 'geometric':
            avg = np.exp(np.mean(np.log(paths), axis=1))
        elif averaging == 'arithmetic':
            avg = np.mean(paths, axis=1)
        else:
            raise ValueError(f"Unknown averaging: {averaging}")

        payoffs = np.where(avg >= K, payoff, 0.0)
        discounted = np.exp(-r * T) * payoffs
        price = float(np.mean(discounted))
        se = float(np.std(discounted, ddof=1) / np.sqrt(n_ref))
        return price, se

    @staticmethod
    def barrier_asian_reference(S0, K, T, r, sigma, H, barrier_type,
                                 n_steps, averaging='geometric',
                                 n_ref=2_000_000, seed=999):
        """
        High-N MC reference for a knock-out Asian barrier option with
        the barrier on the (geometric or arithmetic) average.

        Payoff: (S_T - K)^+ * 1{ A > H }  for 'down-out' (barrier is a
        lower bound on the average).
        """
        dt = T / n_steps
        drift = (r - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt)

        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((n_ref, n_steps))
        log_paths = np.cumsum(drift + diffusion * Z, axis=1)
        paths = S0 * np.exp(log_paths)

        if averaging == 'geometric':
            avg = np.exp(np.mean(np.log(paths), axis=1))
        elif averaging == 'arithmetic':
            avg = np.mean(paths, axis=1)
        else:
            raise ValueError(f"Unknown averaging: {averaging}")

        final_prices = paths[:, -1]
        vanilla = np.maximum(final_prices - K, 0)

        if barrier_type == 'down-out':
            payoffs = np.where(avg > H, vanilla, 0.0)
        elif barrier_type == 'up-out':
            payoffs = np.where(avg < H, vanilla, 0.0)
        elif barrier_type == 'down-in':
            payoffs = np.where(avg <= H, vanilla, 0.0)
        elif barrier_type == 'up-in':
            payoffs = np.where(avg >= H, vanilla, 0.0)
        else:
            raise ValueError(f"Unknown barrier_type: {barrier_type}")

        discounted = np.exp(-r * T) * payoffs
        price = float(np.mean(discounted))
        se = float(np.std(discounted, ddof=1) / np.sqrt(n_ref))
        return price, se

    # ------------------------------------------------------------------
    # Payoff computation
    # ------------------------------------------------------------------

    def price_option_mc(self, stock_paths: np.ndarray) -> np.ndarray:
        """
        Compute option payoffs (undiscounted) from stock paths.
        """
        if self.option_type in ['call', 'put']:
            final_prices = stock_paths[:, -1]
            if self.option_type == 'call':
                payoffs = np.maximum(final_prices - self.K, 0)
            else:
                payoffs = np.maximum(self.K - final_prices, 0)

        elif self.option_type == 'digital':
            final_prices = stock_paths[:, -1]
            payoffs = np.where(final_prices >= self.K, self.digital_payoff, 0.0)

        elif self.option_type == 'asian':
            if self.asian_averaging == 'geometric':
                avg = np.exp(np.mean(np.log(stock_paths), axis=1))
            else:
                avg = np.mean(stock_paths, axis=1)
            payoffs = np.maximum(avg - self.K, 0)

        elif self.option_type == 'asian_floating':
            final_prices = stock_paths[:, -1]
            geo_avg = np.exp(np.mean(np.log(stock_paths), axis=1))
            payoffs = np.maximum(final_prices - geo_avg, 0)

        elif self.option_type == 'digital_asian':
            if self.asian_averaging == 'geometric':
                avg = np.exp(np.mean(np.log(stock_paths), axis=1))
            else:
                avg = np.mean(stock_paths, axis=1)
            payoffs = np.where(avg >= self.K, self.digital_payoff, 0.0)

        elif self.option_type == 'barrier':
            if self.barrier is None:
                final_prices = stock_paths[:, -1]
                payoffs = np.maximum(final_prices - self.K, 0)
            else:
                H = self.barrier
                final_prices = stock_paths[:, -1]

                if self.barrier_type == 'down-out':
                    min_prices = np.min(stock_paths, axis=1)
                    knocked_out = min_prices <= H
                    payoffs = np.where(~knocked_out,
                                       np.maximum(final_prices - self.K, 0), 0.0)
                elif self.barrier_type == 'up-out':
                    max_prices = np.max(stock_paths, axis=1)
                    knocked_out = max_prices >= H
                    payoffs = np.where(~knocked_out,
                                       np.maximum(final_prices - self.K, 0), 0.0)
                elif self.barrier_type == 'down-in':
                    min_prices = np.min(stock_paths, axis=1)
                    knocked_in = min_prices <= H
                    payoffs = np.where(knocked_in,
                                       np.maximum(final_prices - self.K, 0), 0.0)
                elif self.barrier_type == 'up-in':
                    max_prices = np.max(stock_paths, axis=1)
                    knocked_in = max_prices >= H
                    payoffs = np.where(knocked_in,
                                       np.maximum(final_prices - self.K, 0), 0.0)
                else:
                    raise ValueError(f"Unknown barrier type: {self.barrier_type}")

        elif self.option_type == 'barrier_asian':
            if self.barrier is None:
                raise ValueError("Barrier level required for barrier_asian.")
            H = self.barrier
            if self.asian_averaging == 'geometric':
                avg = np.exp(np.mean(np.log(stock_paths), axis=1))
            else:
                avg = np.mean(stock_paths, axis=1)
            final_prices = stock_paths[:, -1]
            vanilla = np.maximum(final_prices - self.K, 0)

            if self.barrier_type == 'down-out':
                payoffs = np.where(avg > H, vanilla, 0.0)
            elif self.barrier_type == 'up-out':
                payoffs = np.where(avg < H, vanilla, 0.0)
            elif self.barrier_type == 'down-in':
                payoffs = np.where(avg <= H, vanilla, 0.0)
            elif self.barrier_type == 'up-in':
                payoffs = np.where(avg >= H, vanilla, 0.0)
            else:
                raise ValueError(
                    f"barrier_asian only supports 'down-out', 'up-out', "
                    f"'down-in', 'up-in'. Got '{self.barrier_type}'."
                )
        else:
            raise ValueError(f"Unknown option type: {self.option_type}")

        return payoffs

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def estimate_price(self, payoffs: np.ndarray) -> Tuple[float, float]:
        """Discount and return (price, std_error)."""
        discounted = np.exp(-self.r * self.T) * payoffs
        price = float(np.mean(discounted))
        se = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
        return price, se

    def uses_terminal_value_only(self) -> bool:
        """True if payoff depends only on S_T."""
        return self.option_type in ('call', 'put', 'digital')

    def payoff_from_terminal(self, Z: np.ndarray) -> np.ndarray:
        """
        Compute payoff directly from terminal-value normals.
        Correct for call / put / digital options.
        """
        if Z.shape[1] != 1:
            raise ValueError(
                f"payoff_from_terminal expects Z of shape (n, 1), got {Z.shape}."
            )
        Z1 = Z[:, 0]
        log_ST = (np.log(self.S0)
                  + (self.r - 0.5 * self.sigma**2) * self.T
                  + self.sigma * np.sqrt(self.T) * Z1)
        S_T = np.exp(log_ST)

        if self.option_type == 'call':
            return np.maximum(S_T - self.K, 0.0)
        elif self.option_type == 'put':
            return np.maximum(self.K - S_T, 0.0)
        elif self.option_type == 'digital':
            return np.where(S_T >= self.K, self.digital_payoff, 0.0)
        else:
            raise ValueError(f"payoff_from_terminal not valid for '{self.option_type}'")

    def get_param_summary(self) -> dict:
        summary = {
            'S0': self.S0, 'K': self.K, 'T': self.T, 'r': self.r,
            'sigma': self.sigma, 'n_simulations': self.n_simulations,
            'n_steps': self.n_steps, 'option_type': self.option_type,
            'closed_form_price': self.closed_form_price,
        }
        if self.barrier is not None:
            summary['barrier'] = self.barrier
            summary['barrier_type'] = self.barrier_type
        if self.option_type in ('asian', 'digital_asian', 'barrier_asian'):
            summary['asian_averaging'] = self.asian_averaging
        if self.option_type in ('digital', 'digital_asian'):
            summary['digital_payoff'] = self.digital_payoff
        return summary