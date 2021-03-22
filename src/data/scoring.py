import numpy as np


scorers = {}


def register_scorer(name):
    """Decorator for classifier scorers"""
    def register_scorer_cls(cls):
        if name in scorers:
            raise ValueError(f"Cannot register duplicate scorer ({name})")
        if not issubclass(cls, BaseClassifierScorer):
            raise ValueError(
                f"Model ({name}: {cls}) must extend Datascorer"
            )
        scorers[name] = cls
        cls._name = name
        return cls

    return register_scorer_cls


class BaseClassifierScorer(object):
    _name = "base"

    def __call__(self, predictions, labels):
        raise NotImplementedError()

    @property
    def name(cls):
        return cls._name


@register_scorer("accuracy")
class Accuracy(BaseClassifierScorer):

    def __call__(self, predictions, labels):
        return (predictions == labels).mean()


@register_scorer("f1")
class F1(BaseClassifierScorer):

    def __call__(self, predictions, labels):
        # True positives
        tp = np.logical_and(predictions == 1, labels == 1).sum()
        # Precision
        P = tp / (predictions == 1).sum()
        # Recall
        R = tp / (labels == 1).sum()
        # F-score
        return 2 * P * R / (P + R)


@register_scorer("matthews")
class Matthews(BaseClassifierScorer):

    def __call__(self, predictions, labels):
        # True/False positives/negatives
        tp = np.logical_and(predictions == 1, labels == 1).sum()
        fp = np.logical_and(predictions == 1, labels == 0).sum()
        tn = np.logical_and(predictions == 0, labels == 0).sum()
        fn = np.logical_and(predictions == 0, labels == 1).sum()
        # Correlation coefficient
        m = (tp * tn) - (fp * fn)
        m /= np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-20

        return m
