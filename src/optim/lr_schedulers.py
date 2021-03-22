# from torch.optim.lr_scheduler import MyLambdaLR
lr_schedulers = {}


def register_lr_scheduler():

    def register_lr_scheduler_fn(fn):
        name = fn.__name__
        if name in lr_schedulers:
            raise ValueError(
                f"Cannot register duplicate lr_scheduler ({name})"
            )
        if not hasattr(fn, "__call__"):
            raise ValueError("An lr_scheduler factory should be a function")
        lr_schedulers[name] = fn
        return fn

    return register_lr_scheduler_fn


class MyLambdaLR(object):
    def __init__(self, optimizer, update_fn):
        self.optimizer = optimizer
        self.update_fn = update_fn
        self._inner_step = 0

    def step(self, override_step=None):
        if override_step is None:
            self._inner_step += 1
            override_step = self._inner_step
        lr = self.update_fn(override_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


@register_lr_scheduler()
def constant(optimizer, lr0, n_steps, warmup):
    def constant_fn(step):
        return lr0
    return MyLambdaLR(optimizer, constant_fn)


@register_lr_scheduler()
def linear_decay(optimizer, lr0, n_steps, warmup):
    def decay_fn(step):
        return lr0*max((n_steps - step) / n_steps, 0)
    return MyLambdaLR(optimizer, decay_fn)


def get_lr_scheduler(name, optimizer, lr0, n_steps, warmup=0):
    """Return learning rate scheduler by name

    Args:
        name (str): Scheduler name
        optimizer (torch.optim.Optimizer): Optimizer
        step (int): variable tracking the step
        n_steps (int): Total number of steps
        warmup (int, optional): Warmup steps. Defaults to 0.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Scheduler
    """
    if name in lr_schedulers:
        lr_scheduler_factory = lr_schedulers[name]
        return lr_scheduler_factory(optimizer, lr0, n_steps, warmup)
    else:
        raise ValueError(f"Unknown lr_scheduler {name}")
