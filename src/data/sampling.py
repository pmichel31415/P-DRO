import numpy as np
import torch as th
from torch.utils.data import Sampler, RandomSampler
from itertools import islice
from .text_dataset import TextDataset


class MixtureSampler(Sampler):
    """This object samples from a mixture of datastreams

    At each timestep it samples a data source according to weights
    and returns an example from the corresonding dataset.

    Stopping conditions are as follows:

    - Subsample: stop whenever any dataset reaches its end
    - Upsample: stop whenever all dataset have reached their end.
        If any dataset runs out before that, it is just restarted.
    - Total: Like Upsample, but stop when the number of iterations
        equals the total number of samples in all datasets. This is
        to make the number of samples in an epoch equivalent to that
        of uniform sampling
    - Bottomless: Just restart any dataset that runs out
        (WARNING: this can lead to an infinite loop).

    The main feature of this is that we can dynamically change the mixture
    weights
    """

    def __init__(self, datasets, weights, stop_condition="upsample"):
        self.k = len(datasets)
        if self.k != len(weights):
            raise ValueError(
                f"Number of datasets ({self.k}) and mixture weights "
                f"({len(weights)}) are different"
            )
        self.datasets = datasets
        self.tot_samples = sum(len(dataset) for dataset in datasets)
        self.dataset_lengths = np.asarray([len(D) for D in datasets])
        self.dataset_offsets = np.cumsum(self.dataset_lengths)
        self.dataset_offsets -= self.dataset_lengths
        self.weights = weights
        self.iters = None
        self.ran_out = [False] * self.k
        self.stop_condition = stop_condition
        self.__iter__()

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, new_weights):
        new_weights = np.asarray(new_weights)
        # Check size
        if len(new_weights) != len(self.datasets):
            raise ValueError(
                f"{len(new_weights)} weights were provided for sampling from "
                f"a mixture of {len(self.datasets)} datasets"
            )
        # Check values
        if any(new_weights < 0):
            raise ValueError("Mixture weights must be positive")
        # What if we get non-zero weight for an empty dataset?
        if any(np.logical_and(self.dataset_lengths == 0, new_weights > 0)):
            print("WARNING: weights of empty datasets automatically set to 0")
            new_weights[self.dataset_lengths == 0] = 0
        # Renormalize
        new_weights /= new_weights.sum()
        # Set the new weights
        self._weights = new_weights

    @property
    def stop_condition(self):
        return self._stop_condition

    @stop_condition.setter
    def stop_condition(self, stop_condition):
        if stop_condition not in {"upsample", "subsample", "total", "bottomless"}:
            raise ValueError(f"Unknown stop condition {stop_condition}.")
        self._stop_condition = stop_condition

    def __iter__(self):
        # Initialize all samplers
        # (we handle empty datasets by ignoring them: they will never be
        # sampled from anyway)
        self.samplers_iters = [
            iter(RandomSampler(dataset)) if len(dataset) > 0 else None
            for dataset in self.datasets
        ]
        self.ran_out = [False] * self.k
        self._n_sampled = 0
        return self

    def __len__(self):
        return self.tot_samples

    def __next__(self):
        # Stop by number of samples
        if self._n_sampled >= self.tot_samples and self.stop_condition == "total":
            raise StopIteration()
        # Sample the component
        mix_idx = np.random.choice(self.k, p=self.weights)
        # Count the sample
        self._n_sampled += 1
        try:
            output = next(self.samplers_iters[mix_idx])
        except StopIteration:
            # Keep track of which iterator has run out at least once
            self.ran_out[mix_idx] = True
            # What happens now depends on the stopping condition
            if self.stop_condition == "subsample":
                # Either stop as soon as one of the iterators runs out
                raise StopIteration()
            elif all(self.ran_out) and self.stop_condition == "upsample":
                # If oversampling, stop when the last iterator has run out
                raise StopIteration()
            else:
                # Otherwise just restart the iterator
                self.samplers_iters[mix_idx] = iter(RandomSampler(
                    self.datasets[mix_idx]
                ))
                # Return the next item
                output = next(self.samplers_iters[mix_idx])
        # We need to offset the index so that the appropriate sample is drawn
        # From the ConcatDataset
        return output + self.dataset_offsets[mix_idx]


INFTY = 99999999


class ByTokensSampler(Sampler):
    """Samples batches from a dataset such that every batch contains at most
    `max_samples` samples and `max_tokens` tokens"""

    def __init__(
        self,
        dataset,
        max_samples,
        max_tokens=None,
        shuffle=False,
        replacement=False,
        num_samples=None,
        other_sampler=None,
    ):
        if not isinstance(dataset, (TextDataset, th.utils.data.ConcatDataset)):
            raise ValueError(
                f"Can only sample by tokens from an instance of TextDataset,"
                f" got {dataset.__class__.__name__} instead."
            )
        self.lengths = [len(f.input_ids) for f in dataset]
        self.shuffle = shuffle
        self.max_samples = max_samples
        self.max_tokens = INFTY if max_tokens is None else max_tokens
        self.batch_idxs = None
        self.replacement = replacement
        self.num_samples = num_samples
        self.other_sampler = other_sampler
        self.determine_order()

    def determine_order(self):
        # Total number of examples
        N = len(self.lengths)
        # First shuffle (or not)
        if self.other_sampler is not None:
            # Sample order from another sampler. This is a hack to get
            # by token sampling to work with weighted sampling
            other_sampler_itr = islice(self.other_sampler, N)
            idxs = th.LongTensor([idx for idx in other_sampler_itr])
        elif self.shuffle and not self.replacement:
            # Sample without replacement
            idxs = th.randperm(N).numpy()
            # If sampling with replacement, sample indices with replacement
        elif self.shuffle and self.replacement:
            n_idxs = N
            # Determine number of indices to sample based on max_samples
            # and batch size
            if self.num_samples is not None:
                n_idxs = self.num_samples*self.max_samples
            idxs = th.randint(low=0, high=N, size=(n_idxs,)).numpy()
        else:
            idxs = th.arange(len(self.lengths)).numpy()
        # Initialize batches
        self.batch_idxs = []
        current_batch = []
        # Running statistics of the batch
        bsz = longest_tokens = 0
        # Iterate over the indices
        for idx in idxs:
            # Update number of tokens. Keeping padding in mind, the total
            # number of tokens will be the batch size multipled by the size
            # of the longest sample
            next_n_tokens = max(longest_tokens, self.lengths[idx]) * (bsz+1)
            # Stop if the batch is too big
            if bsz+1 > self.max_samples or next_n_tokens > self.max_tokens:
                # If the current batch is empty, this mean that this
                # particular sample is just too damn long: ignore it
                if len(current_batch) == 0:
                    print(
                        f"Sample #{idx} is too long ({next_n_tokens}) for the"
                        f" requested number of tokens per batch "
                        f"({self.max_tokens})", flush=True
                    )
                    # Reset current batch
                    current_batch = []
                    bsz = longest_tokens = 0
                    continue
                else:
                    # Add to batch list
                    self.batch_idxs.append(current_batch)
                    # Reset current batch
                    current_batch = []
                    bsz = longest_tokens = 0
            # Keep adding to the batch
            current_batch.append(idx)
            # Update stats
            bsz = len(current_batch)
            longest_tokens = max(self.lengths[idx] for idx in current_batch)

        # Handle the last one
        if len(current_batch) > 0:
            self.batch_idxs.append(current_batch)

        # If sampling with replacement with a fixed number of samples,
        # we might have overestimated the number of examples (if some examples
        # are too long we could have prduced more batches out of them).
        # This step takes care of this.
        if self.replacement and self.num_samples is not None:
            self.batch_idxs = self.batch_idxs[:self.num_samples]

        # Check
        for i, batch_idxs in enumerate(self.batch_idxs):
            bsz = len(batch_idxs)
            max_length = max(self.lengths[idx] for idx in batch_idxs)
            tot_tokens = max_length * bsz
            if tot_tokens > self.max_tokens:
                raise ValueError(
                    f"Batch {i} of size {bsz}x{max_length} has "
                    f"{tot_tokens}>{self.max_tokens} tokens"
                )

    def __len__(self):
        return len(self.batch_idxs)

    def __iter__(self):
        self.determine_order()
        return iter(self.batch_idxs)
