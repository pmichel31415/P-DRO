import os
import numpy as np
from collections import defaultdict
from transformers import DataProcessor
from .text_dataset import ClassificationExample


eos_chars = [".", "?", "!", ","]


def add_full_stop(line):
    """Add a full stop to a line only if it doesn'e already end with
    punctuation"""
    if any(line.endswith(char) for char in eos_chars):
        return line
    else:
        return line + "."


def parse_xml(lines):
    """Parse the ugly xml format and return the bare minimum:

    Review (title + text)
    Rating
    """
    in_content = in_label = False
    review = []
    label = 0
    data = []
    for line in lines:
        line = line.strip()
        if line == "</review_text>" or line == "</title>":
            in_content = False
            if line == "</review_text>":
                # Finalize review: remove tabs (we will save to tsv)
                text = " ".join(review)
                text = text.replace("\t", " ")
                data.append({"label": label, "text": text})
                review = []
        if in_content:
            review.append(add_full_stop(line))
        if line == "<review_text>" or line == "<title>":
            in_content = True
        if in_label:
            label = float(line)
            # Binarize label
            label = 0 if label < 3 else 1
            in_label = False
        if line == "<rating>":
            in_label = True
    return data


def parse_amazon_multi_domains(root_path, out_file="all_domains.tsv"):
    """Read in all domains, binarize them and return a dictionary containing
    all the data"""
    full_data = []
    original_path = os.path.join(root_path, "sorted_data")
    for domain in os.listdir(original_path):
        in_file = os.path.join(original_path, domain, "all.review")
        # Ignore other folders
        if not os.path.isfile(in_file):
            print(f"Didn't find file {in_file}")
            continue
        # Otherwise parse this file
        print(f"Reading {in_file}")
        with open(in_file, "r", encoding="latin-1") as f:
            data = parse_xml([line for line in f])
        # Add domain info
        for x in data:
            x["domain"] = domain
        # Append to the full dataset
        full_data.extend(data)
    # Print to file
    save_tsv(os.path.join(root_path, out_file), full_data)
    return full_data


def save_tsv(filename, data):
    """Save to a tsv file with format [domain]\t[label]\t[text]"""
    with open(filename, "w", encoding="utf-8") as f:
        for x in data:
            fields = [x["domain"], str(x["label"]), x["text"]]
            print("\t".join(fields), file=f)


def load_amazon_tsv(root_path):
    """Load tsv formatted data"""
    data = []
    with open(os.path.join(root_path, "all_domains.tsv"), "r") as tsv:
        for line in tsv:
            domain, label, text = line.strip().split("\t")
            sample = {"domain": domain, "label": int(label), "text": text}
            data.append(sample)
    return data


def split_amazon(data, rng=None, verbose=False, balance=True, n_test=200):
    """Split the Amazon data by domains"""
    # Custom rng for reproducibility
    rng = np.random if rng is None else rng
    idxs_by_domain = defaultdict(lambda: [])
    for idx, x in enumerate(data):
        idxs_by_domain[x["domain"]].append(idx)
    # Balance labels
    if balance:
        for domain, idxs in idxs_by_domain.items():
            neg_idxs = [idx for idx in idxs if data[idx]["label"] == 0]
            pos_idxs = [idx for idx in idxs if data[idx]["label"] == 1]
            # Subsample
            subsample_size = min(len(neg_idxs), len(pos_idxs))
            neg_idxs = rng.choice(neg_idxs, size=subsample_size, replace=False)
            pos_idxs = rng.choice(pos_idxs, size=subsample_size, replace=False)
            idxs_by_domain[domain] = np.concatenate([neg_idxs, pos_idxs])

    # Sort domains by data size
    order_by_size = list(sorted(
        idxs_by_domain.keys(),
        key=lambda x: len(idxs_by_domain[x])
    ))
    # First five: test only
    test_only = set(order_by_size[:10])
    # Next five: test and validation
    valid_and_test_only = set(order_by_size[10:15])
    # Rest: train, valid and test
    train_valid_test = set(order_by_size[15:])
    # Now do the splits
    train_idxs = {}
    valid_idxs = {}
    test_idxs = {}
    for domain, idxs in idxs_by_domain.items():
        # Random order for splitting
        random_idxs = rng.permutation(idxs)
        # Announce domain
        if verbose:
            print(f"Domain \"{domain}\" (total={len(idxs)}):")
        # Now it depends on the domain
        if domain in test_only:
            if verbose:
                print(f" - Test: {len(random_idxs)}", flush=True)
            # All the data goes to the test set
            test_idxs[domain] = random_idxs
        elif domain in valid_and_test_only:
            if verbose:
                print(f" - Valid: {len(random_idxs[-2*n_test:-n_test])}")
                print(f" - Test: {len(random_idxs[-n_test:])}",
                      flush=True)
            # n_test examples go to test set, the rest goes to the dev set
            valid_idxs[domain] = random_idxs[:-n_test]
            test_idxs[domain] = random_idxs[-2*n_test:-n_test]
        elif domain in train_valid_test:
            if verbose:
                print(f" - Train: {len(random_idxs[:-2*n_test])}")
                print(f" - Valid: {len(random_idxs[-2*n_test:-n_test])}")
                print(f" - Test: {len(random_idxs[-n_test:])}", flush=True)
            # n_test examples go to dev and test sets each,
            # the rest goes to the training set
            train_idxs[domain] = random_idxs[:-2*n_test]
            valid_idxs[domain] = random_idxs[-2*n_test:-n_test]
            test_idxs[domain] = random_idxs[-n_test:]
    # Gather the final data
    train = [data[idx] for domain in train_idxs for idx in train_idxs[domain]]
    valid = [data[idx] for domain in valid_idxs for idx in valid_idxs[domain]]
    test = [data[idx] for domain in test_idxs for idx in test_idxs[domain]]
    # Final split sizes
    if verbose:
        print("Total:")
        print(f" - Train: {len(train)}")
        print(f" - Valid: {len(valid)}")
        print(f" - Test: {len(test)}", flush=True)

    return train, valid, test


def test():
    in_file = "/projects/tir4/users/pmichel1/amazon-multi-domain/sorted_data/gourmet_food/all.review"  # noqa
    with open(in_file, "r", encoding="latin-1") as f:
        data = parse_xml([line for line in f])
    print(data[:10])


class AmazonByDomainExample(ClassificationExample):
    """A MultiNLI example contains information on genre
    and the presence of negation"""

    def __init__(self, guid, text_a, label, domain):
        super(AmazonByDomainExample, self).__init__(guid, text_a, None, label)
        # Domain
        self.domain = domain

    @property
    def attributes(self):
        return {"domain": self.domain}


class AmazonByDomainProcessor(DataProcessor):
    """Processor for Amazon by domain"""

    def get_train_examples(self, data_dir, domain=None):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        return self._create_examples(tsv, "train", domain=domain)

    def get_balanced_dev_examples(self, data_dir, domain=None):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "balanced_valid.tsv"))
        return self._create_examples(tsv, "balanced_valid", domain=domain)

    def get_imbalanced_dev_examples(self, data_dir, domain=None):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "imbalanced_valid.tsv"))
        return self._create_examples(tsv, "imbalanced_valid", domain=domain)

    def get_dev_examples(self, data_dir, domain=None):
        return self.get_imbalanced_dev_examples(data_dir, domain)

    def get_dev_full_examples(self, data_dir, domain=None):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "valid.tsv"))
        return self._create_examples(tsv, "valid", domain=domain)

    def get_test_examples(self, data_dir, domain=None):
        """See base class."""
        tsv = self._read_tsv(os.path.join(data_dir, "test.tsv"))
        return self._create_examples(tsv, "test", domain=domain)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, domain=None):
        """Creates examples for the training and dev sets."""
        examples = []
        for (idx, line) in enumerate(lines):
            d, label, text = line
            if domain is not None and d != domain:
                continue
            examples.append(
                AmazonByDomainExample(
                    guid=f"{set_type}-{idx}",
                    text_a=text,
                    label=label,
                    domain=d,
                )
            )
        return examples
