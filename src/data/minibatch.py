#!/usr/bin/env python3
import torch as th


class BaseMiniBatch(object):
    """Base class for minibatch"""

    @property
    def inputs(self):
        """Input variables (typically word ids/pixel tensor)"""
        raise NotImplementedError()

    @property
    def attributes(self):
        """Attributes (metadata that is not an argument of the model)"""
        raise NotImplementedError()

    @attributes.setter
    def attributes(self, attributes):
        """Attributes are mutable (as opposed to features)"""
        raise NotImplementedError()

    @property
    def outputs(self):
        """Output variable (label, etc...)"""
        raise NotImplementedError()

    @property
    def size(self):
        """Batch size"""
        raise NotImplementedError()

    def __len__(self):
        """Number of tensors composing the batch"""
        raise NotImplementedError()

    def to(self, device):
        """Send batch to device"""
        raise NotImplementedError()

    def slice(self, idxs):
        """Select indices from batch"""
        raise NotImplementedError()

    @property
    def device(self):
        """Get batch device"""
        raise NotImplementedError()

    def _getitem_single_idx(self, idx):
        """What __getitem__ will return for a single index"""
        raise NotImplementedError()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._getitem_single_idx(idx)
        elif isinstance(idx, list):
            return [self._getitem_single_idx(i) for i in idx]
        elif isinstance(idx, slice):
            return [self._getitem_single_idx(i)
                    for i in range(*idx.indices(len(self)))]

    def __iter__(self):
        raise NotImplementedError()


class TupleMiniBatch(BaseMiniBatch):
    """Glorified tuple to handle minibatches from different tasks"""

    def __init__(self, elements, attributes=None):
        self.elements = list(elements)
        # Attributes
        self._attributes = {}
        if attributes is not None:
            self._attributes = dict(attributes)
        # Default device is cpu
        self._device = th.device("cpu")
        # If there is any tensor in the batch set the device accordingly
        for elem in self.elements:
            if isinstance(elem, th.Tensor):
                self._device = elem.device
        for v in self._attributes.values():
            if isinstance(v, th.Tensor):
                self._device = v.device

    @property
    def inputs(self):
        """Input variables (typically word ids/pixel tensor)"""
        return self.elements[:-1]

    @property
    def attributes(self):
        """Attributes (metadata that is not an argument of the model)"""
        return self._attributes

    @attributes.setter
    def attributes(self, attributes):
        """Attributes are mutable (as opposed to features)"""
        self._attributes = attributes

    @property
    def outputs(self):
        """Input variables (typically word ids/pixel tensor)"""
        return self.elements[-1]

    @property
    def size(self):
        """Batch size"""
        return len(self.outputs)

    def __len__(self):
        """Number of tensors composing the batch"""
        return len(self.elements)

    def slice(self, idxs):
        """Select indices from batch"""
        sub_elements = [elem[idxs] if hasattr(elem, "__getitem__") else elem
                        for elem in self]
        sub_attributes = {k: v[idxs] if hasattr(v, "__getitem__") else v
                          for k, v in self._attributes.items()}
        return TupleMiniBatch(sub_elements, attributes=sub_attributes)

    def to(self, device):
        """Send batch to device"""
        return TupleMiniBatch(
            [elem.to(device) if isinstance(elem, th.Tensor) else elem
             for elem in self],
            attributes={
                k: v.to(device) if isinstance(v, th.Tensor) else v
                for k, v in self._attributes.items()
            },
        )

    @property
    def device(self):
        """Get batch device"""
        return self._device

    def _getitem_single_idx(self, idx):
        """What __getitem__ will return for a single index"""
        return self.elements[idx]

    def __iter__(self):
        return iter(self.elements)


def default_collate(*args):
    return TupleMiniBatch(th.utils.data.dataloader.default_collate(*args))


def test():
    one, two, three, four = [th.LongTensor([i]) for i in [1, 2, 3, 4]]
    batch = TupleMiniBatch((one, two, three, four, 5))
    # Access all elements
    for idx in range(4):
        print(batch[idx])
    print(batch[1:2])
    print(batch[:])
    print(batch[-1])
    print(batch[1:2])
    print(batch[[0, 3]])
    batch = TupleMiniBatch((one, two, three, four, 5),
                           attributes=(one, two, three, four, 5))
    # Access all elements
    for idx in range(4):
        print(batch[idx])
    print(batch[1:2])
    print(batch[:])
    print(batch[-1])
    print(batch[1:2])
    print(batch[[0, 3]])
    print(batch.attributes)
    print(batch.to("cpu").attributes)


if __name__ == "__main__":
    test()
