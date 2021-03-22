
class Repeating(object):
    """Make a dataloader repeating"""

    def __init__(self, dataloader, max_steps=None):
        self.dataloader = dataloader
        self.itr = None
        self.max_steps = max_steps
        self.step = 0

    def __iter__(self):
        self.step = 0
        self.itr = iter(self.dataloader)
        return self

    def __next__(self):
        # When to stop
        if self.max_steps is not None and self.step >= self.max_steps:
            raise StopIteration()
        # Otherwise get next element from iterator
        try:
            elem = next(self.itr)
        except StopIteration:
            # Just start over
            self.itr = iter(self.dataloader)
            elem = next(self.itr)
        return elem
