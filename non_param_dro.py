#!/usr/bin/env python3
"""
Main program
"""
import numpy as np
import torch as th
import scipy.optimize

import abc
MIN_REL_DIFFERENCE = 1e-5


def bisection_search(objective, min_val, max_val, xtol=1e-5, maxiter=100):

    # Check boundary conditions
    if objective(min_val) * objective(max_val) >= 0:
        # In this case the result lies outside the interval
        if np.abs(objective(min_val)) < np.abs(objective(max_val)):
            # To the left
            root = min_val
        else:
            # Or it lies to the right
            root = max_val
    else:
        # the result lies inside the interval, use the bisection method
        # (=binary search) to find it
        root, results = scipy.optimize.bisect(
            objective, min_val, max_val, xtol=xtol, maxiter=maxiter,
            full_output=True, disp=False,
        )
        if not results.converged:
            print("Bisect didn't converge")

    return root


class NonParametricAdversary(abc.ABC):

    def is_valid_response(self, q: np.ndarray):
        raise NotImplementedError()

    def best_response(self, losses: th.Tensor):
        raise NotImplementedError()


class KLConstrainedAdversary(NonParametricAdversary):
    """KL constrained adversary

    Args:
        kappa (float): KL bound
        log_tau_min (float, optional): Minimum value to check for the log
            temperature. Defaults to -10.
        log_tau_max (float, optional): Maximum value to check for the log
            temperature. Defaults to 10.
    """

    def __init__(
        self, kappa: float, log_tau_min: float = -10, log_tau_max: float = 10
    ) -> None:
        super().__init__()
        self.kappa = kappa
        self.log_tau_min = log_tau_min
        self.log_tau_max = log_tau_max

    def find_optimal_tau(self, losses: np.ndarray):
        """Find \\tau^* such that KL(q^*_\\tau^* || p) = \\kappa

        Heuristically we've found that values of \\tau can be very small
        (<10^2) or sometimes big (10^2). Therefore, searching for \\log_10
        \\tau^* is marginally faster since the values taken by \\tau^* are more
        evenly spread out on the log scale


        Args:
            losses (np.ndarray): sample losses

        Returns:
            float: The optimal \\tau
        """

        def kl_delta(log_tau):
            tau = 10 ** log_tau
            log_q_star_ = losses / tau
            log_q_star_ -= log_q_star_.max()
            log_q_star = log_q_star_ - np.log(np.mean(np.exp(log_q_star_)))
            return (np.exp(log_q_star) * log_q_star).mean() - self.kappa

        log_tau_star = bisection_search(
            kl_delta, self.log_tau_min, self.log_tau_max, xtol=1e-5,
            maxiter=100
        )

        return 10 ** (log_tau_star)

    def is_valid_response(self, q: np.ndarray):
        return np.where(q > 0, q*np.log(q), 0).sum() <= self.kappa

    def best_response(self, losses: th.Tensor):
        tau_star = self.find_optimal_tau(losses.detach().cpu().numpy())
        return th.softmax(losses / tau_star, dim=0)


class Chi2ConstrainedAdversary(NonParametricAdversary):
    def __init__(
        self, bound: float, eta_min: float = -10, eta_max: float = 10
    ) -> None:
        super().__init__()
        self.bound = bound
        self.eta_min = eta_min
        self.eta_max = eta_max

    def is_valid_response(self, q: np.ndarray):
        return 0.5*((q*len(q)-1)**2).mean() <= self.bound

    def find_optimal_eta(self, losses: np.ndarray):
        """Find \\eta^* such that KL(q^*_\\eta^* || p) = \\kappa


        Args:
            losses (np.ndarray): sample losses

        Returns:
            float: The optimal \\eta
        """

        def chi2_delta(eta):
            q_star_ = np.maximum(1e-12, losses - eta)
            q_star = q_star_ / q_star_.sum()
            m = len(losses)
            return 0.5*((m*q_star - 1)**2).mean() - self.bound

        eta_min = -(1.0 / (np.sqrt(2 * self.bound + 1) - 1)) * losses.max()
        eta_max = losses.max()

        eta_star = bisection_search(
            chi2_delta, eta_min, eta_max, xtol=1e-3, maxiter=100
        )

        return eta_star

    def best_response(self, losses: th.Tensor):
        # If the losses are too close, just return uniform weights
        if (losses.max() - losses.min()) / losses.max() <= MIN_REL_DIFFERENCE:
            return th.ones_like(losses) / len(losses)
        # failsafe for batch sizes small compared to uncertainty set size
        if len(losses) <= 1 + 2 * self.bound:
            out = (losses == losses.max()).float()
            out /= out.sum()
            return out
        # Otherwise find optimal eta
        eta_star = self.find_optimal_eta(losses.detach().cpu().numpy())
        q_star_ = th.relu(losses - eta_star)
        return q_star_ / q_star_.sum()


class CVaRConstrainedAdversary(NonParametricAdversary):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def is_valid_response(self, q: np.ndarray):
        return q.max() <= 1/(len(q)*self.alpha)

    def best_response(self, losses: th.Tensor):
        m = len(losses)
        # We assign the maximum weight (1 / alpha)
        cutoff = int(self.alpha * m)
        surplus = 1.0 - cutoff / (self.alpha * m)
        p = th.zeros_like(losses)
        idx = th.argsort(losses, descending=True)
        p[idx[:cutoff]] = 1.0 / (self.alpha * m)
        if cutoff < m:
            p[idx[cutoff]] = surplus
        return p
