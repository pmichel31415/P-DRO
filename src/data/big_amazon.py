import tqdm
import numpy as np
import json
import re
import os
from hashlib import md5
from utils import Reservoir
rating_regex = re.compile(r"\"overall\": (\d\.0)")


def filter_big_amazon(
    in_file,
    out_file,
    max_length=200,
    max_examples=5000000,
    rng=None,
    balanced=True,
    deduplicate=True,
):
    # Initialize RNG
    if rng is None:
        rng = np.random
    # Reservoirs for sampling both positive and negative examples
    pos_samples = Reservoir(max_examples//2, rng)
    neg_samples = Reservoir(max_examples//2, rng)
    # Hashes for deduplication
    hashes = set()
    duplicates = too_long = neutral = invalid = 0
    # Go ahead
    with open(in_file, "r") as f:
        itr = tqdm.tqdm(enumerate(f), desc="Reading raw json")
        for idx, line in itr:
            # Update progress bar
            percent_rejected = duplicates + too_long + neutral + invalid
            percent_rejected = percent_rejected/max(idx, 1)*100
            itr.set_description(
                f"Reading {os.path.basename(in_file)}, "
                f"rejected {percent_rejected:.1f}%"
                f" (dups={duplicates} too long={too_long} neutral={neutral} "
                f"invalid={invalid})"
            )
            data = json.loads(line)
            if "reviewText" not in data or "overall" not in data:
                invalid += 1
                continue
            # Retrieve the text and label (only thing we need for filtering)
            text = data["reviewText"]
            rating = data["overall"]
            # Unique hash for the review based on text and rating
            review_hash = md5((str(rating) + text).encode()).hexdigest()
            # Skip duplicates
            if deduplicate and review_hash in hashes:
                duplicates += 1
                continue
            # Track hash
            hashes.add(review_hash)
            # Ignore reviews that are too long
            if len(text.split()) > max_length:
                too_long += 1
                continue
            # Rating >3 is positive, <3 is negative, ignore neutral ratings
            if rating > 3:
                pos_samples.add(idx)
            elif rating < 3:
                neg_samples.add(idx)
            else:
                neutral += 1
    # Retrieve selected indices
    pos_idxs = pos_samples.container
    neg_idxs = neg_samples.container
    # Rebalance
    if balanced:
        size = min(len(pos_idxs), len(neg_idxs))
        pos_idxs = rng.choice(pos_idxs, size, replace=False)
        neg_idxs = rng.choice(neg_idxs, size, replace=False)
    idxs = set(pos_idxs)
    idxs.update(neg_idxs)
    # Write the subset to a file
    with open(in_file, "r") as in_f, open(out_file, "w") as out_f:
        # Progress bar
        t = tqdm.tqdm(total=len(idxs), desc="Writing filtered JSON")
        n_written = 0
        for idx, line in enumerate(in_f):
            if idx in idxs:
                print(line.strip(), file=out_f)
                n_written += 1
                t.update(1)


def convert_to_tsv(in_file, out_file):
    with open(in_file, "r") as f, open(out_file, "w") as out_f:
        itr = tqdm.tqdm(enumerate(f), desc="Reading raw json")
        for idx, line in itr:
            data = json.loads(line.strip())
            # Text
            text = data["reviewText"]
            # Binary label
            label = "1" if data["overall"] > 3 else "0"
            # Product ID
            prod_id = data["asin"]
            # Reviewer ID
            rev_id = data["reviewerID"]
            # Print to output file
            print("\t".join([text, label, prod_id, rev_id]), file=out_f)


def test():
    root = "/projects/tir4/users/pmichel1/amazon-multi-domain"
    filter_big_amazon(
        os.path.join(root, "AMAZON_FASHION_5.json"),
        os.path.join(root, "AMAZON_FASHION_5_filtered.json"),
        rng=np.random.RandomState(4328756)
    )
    convert_to_tsv(
        os.path.join(root, "AMAZON_FASHION_5_filtered.json"),
        os.path.join(root, "AMAZON_FASHION_5_filtered.tsv"),
    )


def main():
    rng = np.random.RandomState(20657028)
    root = "/projects/tir4/users/pmichel1/amazon-multi-domain"
    # Create filtered dir
    if not os.path.isdir(os.path.join(root, "big_amazon_filtered")):
        os.mkdir(os.path.join(root, "big_amazon_filtered"))
    # Filter all
    for filename in os.listdir(os.path.join(root, "big_amazon")):
        if filename.endswith(".json"):
            filter_big_amazon(
                os.path.join(root, "big_amazon", filename),
                os.path.join(root, "big_amazon_filtered", filename),
                rng=rng
            )
    # Now to tsv
    for filename in os.listdir(os.path.join(root, "big_amazon_filtered")):
        if filename.endswith(".json"):
            convert_to_tsv(
                os.path.join(root, "big_amazon_filtered", filename),
                os.path.join(root, "big_amazon_filtered",
                             filename[:-4] + "tsv"),
            )


if __name__ == "__main__":
    main()
