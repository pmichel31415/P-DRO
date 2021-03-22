from torch import nn


class ModelWithHead(nn.Module):
    """A model with a classification head."""

    def __init__(self, model, head=None, combination_fn=None):
        super(ModelWithHead, self).__init__()
        self.model = model
        self.head = nn.Identity() if head is None else head
        self.combination_fn = combination_fn

    def forward(self, *args, **kwargs):
        h = self.model(*args, **kwargs)
        if self.combination_fn is None:
            return self.head(h)
        else:
            return self.combination_fn(self.head, h)
