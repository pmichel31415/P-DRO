import os.path
from collections import defaultdict
import numpy as np

from src.data.utils import Reservoir
from src.data.amazon_multi_domain import (
    load_amazon_tsv,
    save_tsv,
    split_amazon,
)


def main():
    root = "/projects/tir4/users/pmichel1/amazon-multi-domain"
    rng = np.random.RandomState(seed=207081993)
    n_test = 1000
    # Parse to file
    # parse_amazon_multi_domains(root)
    # Split
    data = load_amazon_tsv(root)
    train, valid, test = split_amazon(
        data,
        rng=rng,
        verbose=True,
        n_test=n_test,
    )
    save_tsv(os.path.join(root, "train.tsv"), train)
    save_tsv(os.path.join(root, "valid.tsv"), valid)
    save_tsv(os.path.join(root, "test.tsv"), test)

    # Training proportions
    train_ratios = defaultdict(lambda: 0)
    for x in train:
        train_ratios[x["domain"]] += 1
    train_ratios = {d: n/len(train) for d, n in train_ratios.items()}
    print("Training proportions")
    for d, p in sorted(train_ratios.items(), reverse=True, key=lambda x: x[1]):
        print(f" - {d}: {p*100:.1f}%")
    # Imbalanced dev set
    # We assign the maximum amount to the largest domain (=n_test)
    # The we assign to other domains according to their proportions
    # in the training data
    max_share = max(train_ratios.values())
    imbal_reservoirs = {d: Reservoir(int(p * n_test / max_share), rng)
                        for d, p in train_ratios.items()}
    # We will also subsample a smaller dev set that is balanced
    # but the same size as the imbalanced dev set
    n_imbal = sum(r.capacity for r in imbal_reservoirs.values())
    n_domains = len(train_ratios)
    bal_reservoirs = {d: Reservoir(n_imbal/n_domains, rng)
                      for d, p in train_ratios.items()}
    # Add to the reservoirs
    for x in valid:
        if x["domain"] in train_ratios:
            imbal_reservoirs[x["domain"]].add(x)
            bal_reservoirs[x["domain"]].add(x)
    # Concatenate to the final imbalanced set
    imbalanced_valid = [
        x for d, r in imbal_reservoirs.items()
        for x in r.container
    ]
    balanced_valid = [
        x for d, r in bal_reservoirs.items()
        for x in r.container
    ]
    save_tsv(os.path.join(root, "imbalanced_valid.tsv"), imbalanced_valid)
    save_tsv(os.path.join(root, "balanced_valid.tsv"), balanced_valid)


if __name__ == "__main__":
    main()
