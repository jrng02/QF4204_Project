"""
Evaluation metrics for Monte Carlo methods.

Includes standard error, bias, RMSE, VRF (Variance Reduction Factor),
and interpolated VRF following Atanassov & Wang (2008) methodology.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict, List, Optional
import warnings


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for Monte Carlo methods.
    """

    def __init__(self, true_price: float):
        """
        Parameters
        ----------
        true_price : float
            True/reference option price.
        """
        self.true_price = true_price
        self.metrics = {}

    # ------------------------------------------------------------------
    # Basic error metrics
    # ------------------------------------------------------------------

    def compute_rmse(self, estimated_prices: np.ndarray) -> float:
        """Root Mean Square Error."""
        return float(np.sqrt(np.mean((estimated_prices - self.true_price) ** 2)))

    def compute_mse(self, estimated_prices: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((estimated_prices - self.true_price) ** 2))

    def compute_mae(self, estimated_prices: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(estimated_prices - self.true_price)))

    def compute_bias(self, estimated_prices: np.ndarray) -> float:
        """Bias = mean estimate - true price."""
        return float(np.mean(estimated_prices) - self.true_price)

    def compute_variance(self, estimated_prices: np.ndarray) -> float:
        """Variance of the estimates."""
        return float(np.var(estimated_prices, ddof=1)) if len(estimated_prices) > 1 else 0.0

    def compute_std_dev(self, estimated_prices: np.ndarray) -> float:
        """Standard deviation of the estimates."""
        return float(np.std(estimated_prices, ddof=1)) if len(estimated_prices) > 1 else 0.0

    def compute_relative_rmse(self, estimated_prices: np.ndarray) -> float:
        """Relative RMSE as a percentage of the true price."""
        rmse = self.compute_rmse(estimated_prices)
        if self.true_price != 0:
            return float(rmse / abs(self.true_price) * 100)
        return np.inf

    # ------------------------------------------------------------------
    # Confidence interval
    # ------------------------------------------------------------------

    def compute_confidence_interval(self, estimated_prices: np.ndarray,
                                    confidence: float = 0.95
                                    ) -> Tuple[float, float]:
        """
        Compute a confidence interval for the mean estimate.

        Parameters
        ----------
        estimated_prices : np.ndarray
            Array of independent price estimates.
        confidence : float
            Confidence level (default 0.95).

        Returns
        -------
        (lower, upper) : Tuple[float, float]
        """
        n = len(estimated_prices)
        if n < 2:
            m = float(np.mean(estimated_prices))
            return m, m
        mean = float(np.mean(estimated_prices))
        std_err = float(np.std(estimated_prices, ddof=1) / np.sqrt(n))
        z_score = float(norm.ppf((1 + confidence) / 2))
        return mean - z_score * std_err, mean + z_score * std_err

    # ------------------------------------------------------------------
    # Convergence rate
    # ------------------------------------------------------------------

    def compute_convergence_rate(self, simulation_sizes: List[int],
                                 errors: List[float]) -> float:
        """
        Estimate the convergence rate alpha from error ~ C * N^{-alpha}.

        Fits a straight line to log(error) vs log(N) and returns -slope.
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            valid_idx = [i for i, e in enumerate(errors) if e > 0]
            if len(valid_idx) < 2:
                return np.nan

            log_sizes = np.log([simulation_sizes[i] for i in valid_idx])
            log_errors = np.log([errors[i] for i in valid_idx])
            slope, _ = np.polyfit(log_sizes, log_errors, 1)

        return float(-slope)

    # ------------------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------------------

    def compute_efficiency(self, price: float, std_error: float,
                           computation_time: float) -> float:
        """
        Efficiency = 1 / (variance * time). Higher is better.
        """
        variance = std_error ** 2
        denom = variance * computation_time
        if np.isfinite(variance) and variance > 1e-25 and np.isfinite(computation_time) and computation_time > 0:
            return float(1.0 / denom)
        return float('nan')

    # ------------------------------------------------------------------
    # Variance Reduction Factor (VRF)
    # ------------------------------------------------------------------

    def compute_vrf(self, mc_prices: np.ndarray,
                    rqmc_prices: np.ndarray) -> float:
        """
        Compute the Variance Reduction Factor (VRF) of a method relative
        to a baseline MC method.

        VRF = Var(MC) / Var(method)

        Both arrays should contain the same number of independent runs.
        Higher is better. A VRF of 100 means the method is 100x more
        efficient than plain MC.

        Parameters
        ----------
        mc_prices : np.ndarray
            Price estimates from the baseline MC method.
        rqmc_prices : np.ndarray
            Price estimates from the method being evaluated.

        Returns
        -------
        vrf : float
        """
        var_mc = np.var(mc_prices, ddof=1)
        var_method = np.var(rqmc_prices, ddof=1)
        if np.isfinite(var_method) and var_method > 0:
            return float(var_mc / var_method)
        # Zero observed variance does not imply zero RMSE; empirical VRF
        # is not estimable from these replications. Never report VRF=0/inf.
        return np.nan

    def compute_vrf_interpolated(self, mc_prices: np.ndarray,
                                 rqmc_prices_by_n: Dict[int, np.ndarray],
                                 n_target: int = 2 ** 20) -> Dict:
        """
        Interpolated VRF following Atanassov & Wang (2008).

        Fits the empirical variance of the RQMC estimator as
            Var_method(n) ~ a0 * n^{-nu}
        using a log-log regression over the observed sample sizes, and
        then computes the VRF at a target sample size n_target:

            VRF(n_target) = Var(MC) / (a0 * n_target^{-nu})

        Parameters
        ----------
        mc_prices : np.ndarray
            Price estimates from the baseline MC method.
        rqmc_prices_by_n : Dict[int, np.ndarray]
            Mapping from sample size N to an array of independent price
            estimates from the method being evaluated.
        n_target : int
            Sample size at which to evaluate the interpolated VRF.

        Returns
        -------
        dict with keys:
            vrf             : float
            nu              : float  (empirical convergence exponent)
            a0              : float  (variance prefactor)
            n_observed      : list   (sample sizes used)
            var_observed    : list   (empirical variances)
        """
        ns = sorted(rqmc_prices_by_n.keys())
        variances = np.array([np.var(rqmc_prices_by_n[n], ddof=1) for n in ns])

        # Filter out invalid variances (zeros or negatives)
        valid = variances > 0
        if valid.sum() < 2:
            return {
                'vrf': np.nan,
                'nu': np.nan,
                'a0': np.nan,
                'n_observed': ns,
                'var_observed': variances.tolist(),
            }

        ns_valid = np.array(ns)[valid]
        var_valid = variances[valid]

        log_n = np.log(ns_valid)
        log_var = np.log(var_valid)
        slope, intercept = np.polyfit(log_n, log_var, 1)
        nu_hat = -slope
        a0_hat = np.exp(intercept)

        var_mc = np.var(mc_prices, ddof=1)
        vrf = var_mc / (a0_hat * (n_target ** (-nu_hat)))

        return {
            'vrf': float(vrf),
            'nu': float(nu_hat),
            'a0': float(a0_hat),
            'n_observed': ns,
            'var_observed': variances.tolist(),
        }

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def compute_all_metrics(self, estimated_prices: np.ndarray,
                            price: float, std_error: float,
                            computation_time: float,
                            mc_prices: Optional[np.ndarray] = None
                            ) -> Dict[str, float]:
        """
        Compute all standard metrics.

        Parameters
        ----------
        estimated_prices : np.ndarray
            Independent price estimates.
        price : float
            Mean price estimate.
        std_error : float
            Standard error of the mean estimate.
        computation_time : float
            Average computation time per run.
        mc_prices : np.ndarray, optional
            If provided, the VRF is computed against this baseline.
            Typically the array of price estimates from plain MC.

        Returns
        -------
        dict
        """
        metrics = {
            'RMSE': self.compute_rmse(estimated_prices),
            'MSE': self.compute_mse(estimated_prices),
            'MAE': self.compute_mae(estimated_prices),
            'Bias': self.compute_bias(estimated_prices),
            'Variance': self.compute_variance(estimated_prices),
            'Std_Dev': self.compute_std_dev(estimated_prices),
            'Standard_Error': float(std_error),
            'Computation_Time': float(computation_time),
            'Efficiency': self.compute_efficiency(price, std_error, computation_time),
            'Price_Estimate': float(price),
            'True_Price': float(self.true_price),
            'Absolute_Error': float(abs(price - self.true_price)),
            'Relative_Error': float(abs(price - self.true_price) / self.true_price * 100)
                              if self.true_price != 0 else np.inf,
            'Relative_RMSE': self.compute_relative_rmse(estimated_prices),
        }

        # Confidence interval
        lower, upper = self.compute_confidence_interval(estimated_prices)
        metrics['CI_Lower'] = float(lower)
        metrics['CI_Upper'] = float(upper)
        metrics['CI_Width'] = float(upper - lower)

        # VRF (only if baseline provided)
        if mc_prices is not None:
            metrics['VRF'] = self.compute_vrf(mc_prices, estimated_prices)
        else:
            metrics['VRF'] = np.nan

        return metrics