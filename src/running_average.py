#!/usr/bin/env python3
"""
Code for keeping track of various running averages
"""
import numpy as np


def logsumexp(log_x):
    """Stable logsumexp for numpy

    Args:
        log_x (np.ndarray): Numpy array

    Returns:
        float: Return log(sum(exp(log_x)))
    """
    max_ = np.max(log_x)
    return max_ + np.log(np.sum(np.exp(log_x-max_)))


class LogRunningAverage(object):
    """Compute the log of a running average over a fixed history of log values

    This saves the previous `N` log values log(x_1), ... log(x_N) and returns
    log((exp(x_1) + ... + exp(x_N) / N)

    Args:
        N (int): Size of the history
    """

    def __init__(self, N):
        self.N = N
        # This memory stores the log values
        self._memory = []

    def __iadd__(self, log_x):
        """Add a value to the memory queue

        Args:
            log_x (float): log of the value to add
        """
        # Add
        self._memory.append(log_x)
        # Pop the first element (FIFO)
        if len(self._memory) > self.N:
            self._memory.pop(0)
        return self

    @property
    def value(self):
        # Compute the logsumexp and substract the log normalizer
        return logsumexp(self._memory) - np.log(len(self._memory))


class ConstantLogRunningAverage(LogRunningAverage):
    """This is a stand-in for LogRunningAverage when we don't want to
    compute a running average and want to return a fixed value.

    Args:
        value (int, optional): Desired value (in log space). Defaults to 0.
    """

    def __init__(self, value=0):
        self._value = value

    def __iadd__(self, log_x):
        return self

    @property
    def value(self):
        return self._value


class LogExponentialRunningAverage(LogRunningAverage):
    """Computes the log of an exponentially decaying average

    As a reminder the exponentially decaying average y_n of n values
    x_1, ... , x_n with decay parameter α is computed recursively as:

    y_n = α * x_n + (1-α) * y_{n-1}

    Here, everything is done in log space (ie the log of the values log(x_i))
    are provided and log(y_n) is returned

    Args:
        decay (float): Decay parameter (a value in [0, 1]). The higher the
        value, the more weight is assigned to recent values
    """

    def __init__(self, decay):
        self.decay = decay
        self._value = None

    def __iadd__(self, x):
        if self._value is None or self.decay == 0:
            self._value = x
        else:
            self._value = np.logaddexp(
                self._value + np.log(self.decay),
                x + np.log(1-self.decay),
            )
        return self

    @property
    def value(self):
        return self._value


def get_log_running_average(norm_k):
    if norm_k is None:
        # No normalization
        return ConstantLogRunningAverage(0)
    elif norm_k >= 1:
        # If the history is >1, fixed size history
        return LogRunningAverage(int(norm_k))
    else:
        # Otherwise, exponential decay
        return LogExponentialRunningAverage(norm_k)
