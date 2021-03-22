#!/usr/bin/env python3
import json

ROOT = "/projects/tir1/users/pmichel1/hatespeech"

# FOUNTA

data = {"train": [], "valid": [], "test": []}
with open(f"{ROOT}/data/founta/exp/label/random/all.jsonlist", "r", encoding="utf-8") as f:
    for idx, l in enumerate(f):
        sample = json.loads(l)
        split = "train"
        if sample["folds"]["0"] == "test":
            split = "test"
        elif sample["folds"]["0"] == 0:
            split = "valid"
        data[split].append(sample)


for split in data:
    with open(f"{ROOT}/data/founta/{split}.tsv", "w", encoding="utf-8") as f:
        print("\t".join(["label", "dialect", "text"]), file=f)
        for sample in data[split]:
            print(
                "\t".join([sample["label"], sample["dialect"], sample["text"]]), file=f)

# DAVIDSON

data = {"train": [], "valid": [], "test": []}
with open(f"{ROOT}/data/davidson/exp/label/random/all.jsonlist", "r", encoding="utf-8") as f:
    for idx, l in enumerate(f):
        sample = json.loads(l)
        split = "train"
        if sample["folds"]["0"] == "test":
            split = "test"
        elif sample["folds"]["0"] == "0":
            split = "valid"
        data[split].append(sample)


for split in data:
    with open(f"{ROOT}/data/davidson/{split}.tsv", "w", encoding="utf-8") as f:
        print("\t".join(["label", "dialect", "text"]), file=f)
        for sample in data[split]:
            print(
                "\t".join([sample["label"], sample["dialect"], sample["text"]]), file=f)
