import numpy as np

_INFINITY = np.inf


def compare_scores(current, previous_best, lower_is_better):
    if lower_is_better:
        return (current < previous_best) or previous_best == _INFINITY
    else:
        return (current > previous_best) or previous_best == 0


class OptimalStopping(object):
    """Handles early stopping"""

    def is_best(self, valid_scores):
        """Returns True if the current model is the best

        Args:
            valid_scores (np.ndarray): 1-D array containing the scores for
                each validation sample (error, loss, reward, etc...)

        """
        raise NotImplementedError()

    def update_best(self, valid_scores):
        raise NotImplementedError()

    @property
    def previous_best(self):
        raise NotImplementedError()

    @property
    def current_score(self):
        raise NotImplementedError()


class AverageStopping(OptimalStopping):
    """Stop based on average accuracy

    Args:
        lower_is_better (bool, optional): Indicate that a lower score is better
            (for eg. perplexity).Defaults to True.
    """

    def __init__(self, lower_is_better=True):
        self.lower_is_better = lower_is_better
        self._previous_best = _INFINITY if lower_is_better else -1e-12
        self._current = self._previous_best

    def is_best(self, valid_scores):
        current_score = valid_scores.mean()
        return compare_scores(
            current_score,
            self._previous_best,
            self.lower_is_better,
        )

    def update_best(self, valid_scores):
        current_score = valid_scores.mean()
        is_best = compare_scores(
            current_score,
            self._previous_best,
            self.lower_is_better,
        )
        self._current = current_score
        if is_best:
            self._previous_best = current_score
        return is_best

    @property
    def previous_best(self):
        return self._previous_best

    @property
    def current_score(self):
        return self._current


class GreedyMinMaxStopping(OptimalStopping):
    """Stop based on min-max accuracy/loss

    Args:
        n_valid (np.ndarray): Number of validation examples
        lower_is_better (bool, optional): Indicate that a lower score is better
        (for eg. perplexity).Defaults to True.
    """

    def __init__(self, n_valid, lower_is_better=True):
        self.lower_is_better = lower_is_better
        worst_score = _INFINITY if lower_is_better else -1e-12
        self._previous_best = np.full(n_valid, worst_score)
        self._current = self._previous_best
        self.all_adv_log_weights = [np.zeros(n_valid)]

    def add_adv_log_weights(self, adv_log_weights):
        self.all_adv_log_weights.append(adv_log_weights)

    def robust_score(self, valid_scores):
        all_log_weights = np.stack(self.all_adv_log_weights, axis=0)
        all_weights = np.exp(all_log_weights - np.log(len(valid_scores)))
        adv_scores = all_weights.dot(valid_scores)
        if self.lower_is_better:
            return adv_scores.max()
        else:
            return adv_scores.min()

    def is_best(self, valid_scores):
        current_score = self.robust_score(valid_scores)
        previous_best_score = self.robust_score(self._previous_best)
        return compare_scores(
            current_score,
            previous_best_score,
            self.lower_is_better,
        )

    def update_best(self, valid_scores):
        # Compute robust accuracy over
        current_score = self.robust_score(valid_scores)
        previous_best_score = self.robust_score(self._previous_best)
        is_best = compare_scores(
            current_score,
            previous_best_score,
            self.lower_is_better,
        )
        self._current = valid_scores.copy()
        if is_best:
            self._previous_best = valid_scores.copy()
        return is_best

    @property
    def previous_best(self):
        return self.robust_score(self._previous_best)

    @property
    def current_score(self):
        return self.robust_score(self._current)


class GroupRobustStopping(OptimalStopping):
    """Stop based on min-max accuracy/loss on a set of groups

    Args:
        group_idxs (np.ndarray): Initial adversarial weights
        lower_is_better (bool, optional): Indicate that a lower score is better
        (for eg. perplexity).Defaults to True.
    """

    def __init__(self, group_idxs, lower_is_better=True):
        self.lower_is_better = lower_is_better
        worst_score = _INFINITY if lower_is_better else -1e-12
        self._previous_best = np.full(len(group_idxs), worst_score)
        self._current_group_scores = self._previous_best.copy()
        self.group_idxs = group_idxs

    def robust_score(self, group_scores):
        if self.lower_is_better:
            return group_scores.max()
        else:
            return group_scores.min()

    def is_best(self, valid_scores):
        group_scores = np.asarray([valid_scores[g_idxs].mean()
                                   for g_idxs in self.group_idxs.values()])
        current_score = self.robust_score(group_scores)
        return compare_scores(
            current_score,
            self.robust_score(self._previous_best),
            self.lower_is_better,
        )

    def update_best(self, valid_scores):
        group_scores = np.asarray([valid_scores[g_idxs].mean()
                                   for g_idxs in self.group_idxs.values()])
        # Compute robust accuracy over
        current_score = self.robust_score(group_scores)
        is_best = compare_scores(
            current_score,
            self.robust_score(self._previous_best),
            self.lower_is_better,
        )
        self._current_group_scores = group_scores.copy()
        if is_best:
            self._previous_best = group_scores.copy()
        return is_best

    @property
    def previous_best(self):
        return self.robust_score(self._previous_best)

    @property
    def current_score(self):
        return self.robust_score(self._current_group_scores)
