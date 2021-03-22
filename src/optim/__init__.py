from .adam import AdamOverride
from .sgd import SGDOverride
from torch.optim import AdamW, RMSprop
from .lr_schedulers import get_lr_scheduler, lr_schedulers

optimizers = {}


def register_optimizer():

    def register_optimizer_fn(fn):
        name = fn.__name__
        if name in optimizers:
            raise ValueError(f"Cannot register duplicate optimizer ({name})")
        if not hasattr(fn, "__call__"):
            raise ValueError("An optimizer factory should be a function")
        optimizers[name] = fn
        return fn

    return register_optimizer_fn


@register_optimizer()
def sgd(parameters, lr, weight_decay, mom):
    """Standard SGD with gradient override"""
    return SGDOverride(
        parameters,
        lr=lr,
        momentum=mom,
        weight_decay=weight_decay,
    )


@register_optimizer()
def rmsprop(parameters, lr, weight_decay, mom):
    return RMSprop(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
    )


@register_optimizer()
def adam(parameters, lr, weight_decay, mom):
    # TODO: add more options
    return AdamOverride(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
    )


@register_optimizer()
def amsgrad(parameters, lr, weight_decay, mom):
    return AdamOverride(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
        amsgrad=True,
    )


@register_optimizer()
def adamw(parameters, lr, weight_decay, mom):
    return AdamW(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
    )


@register_optimizer()
def adamw_amsgrad(parameters, lr, weight_decay, mom):
    return AdamW(
        parameters,
        lr=lr,
        weight_decay=weight_decay,
        amsgrad=True,
    )


def get_optimizer(optim_name, parameters, lr, weight_decay, mom=0):
    if optim_name in optimizers:
        optimizer_factory = optimizers[optim_name]
        return optimizer_factory(parameters, lr, weight_decay, mom)
    else:
        raise ValueError(f"Unknown optimizer {optim_name}")


__all__ = [
    "get_lr_scheduler",
    "get_optimizer",
    "lr_schedulers",
]
