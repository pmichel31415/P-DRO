#!/usr/bin/env python3
"""Adapted from
https://github.com/huggingface/pytorch-pretrained-BERT

Classes to load SuperGLUE classes
"""
import os
import json

from transformers import InputExample, DataProcessor

from .multiple_choice import MCInputExample

superglue_processors = {}


class SuperGlueProcessor(DataProcessor):
    """Processor for Superglue tasks"""

    @classmethod
    def _read_jsonl(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as jsonl_file:
            lines = []
            for line in jsonl_file:
                lines.append(json.loads(line))
            return lines

    def get_train_examples(self, data_dir):
        """See base class."""
        jsonl = self._read_jsonl(os.path.join(data_dir, "train.jsonl"))
        return self._create_examples(jsonl, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        jsonl = self._read_jsonl(os.path.join(data_dir, "val.jsonl"))
        return self._create_examples(jsonl, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        jsonl = self._read_jsonl(os.path.join(data_dir, "test.jsonl"))
        return self._create_examples(jsonl, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data["race_id"])
            article = data["article"]
            for i in range(len(data["answers"])):
                truth = str(ord(data["answers"][i]) - ord("A"))
                question = data["questions"][i]
                options = data["options"][i]

                examples.append(
                    MCInputExample(
                        guid=race_id,
                        question=question,
                        # this is not efficient but convenient
                        contexts=[article, article, article, article],
                        endings=[options[0], options[1],
                                 options[2], options[3]],
                        label=truth,
                    )
                )
        return examples


class BoolQProcessor(SuperGlueProcessor):
    """BoolQ dataset: true/false statements"""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data) in enumerate(lines):
            examples.append(
                InputExample(
                    guid=f"{set_type}-{data['idx']}",
                    text_a=data["passage"],
                    text_b=data["question"],
                    label=str(data["label"]),
                )
            )
        return examples

    def get_labels(self):
        """See base class."""
        return ["False", "True"]


class CBProcessor(SuperGlueProcessor):
    """CommitmentBank dataset: NLI"""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data) in enumerate(lines):
            examples.append(
                InputExample(
                    guid=f"{set_type}-{data['idx']}",
                    text_a=data["premise"],
                    text_b=data["hypothesis"],
                    label=str(data["label"]),
                )
            )
        return examples

    def get_labels(self):
        """See base class."""
        return ["entailment", "neutral", "contradiction"]


class COPAProcessor(SuperGlueProcessor):
    """COPA dataset: true/false statements on cause or effect relationship"""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data) in enumerate(lines):
            question = (
                "What was the cause of this?"
                if data["question"] == "cause"
                else "What happened as a result?"
            )
            examples.append(
                MCInputExample(
                    guid=f"{set_type}-{data['idx']}",
                    # This is just "cause" or "effect"
                    question=question,
                    contexts=[data["premise"], data["premise"]],
                    endings=[data["choice1"], "choice2"],
                    label=str(data["label"]),
                )
            )
        return examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


class MultiRCProcessor(SuperGlueProcessor):
    """MultiRC: questions on a paragraph with multiple possible answers"""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data) in enumerate(lines):
            passage = data["passage"]["text"]
            for Q in data["passage"]["questions"]:
                question = Q["question"]
                for A in Q["answers"]:
                    guid = f"{set_type}-{data['idx']-Q['idx']-A['idx']}"
                    examples.append(
                        InputExample(
                            guid=guid,
                            text_a=passage,
                            text_b=question + " " + A["text"],
                            label=str(A["label"]),
                        )
                    )
        return examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


class ReCoRDProcessor(SuperGlueProcessor):
    """ReCoRD dataset: Cloze questions"""

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data) in enumerate(lines):
            passage = data["passage"]
            ents_pos = [(pos["start"], pos["end"])
                        for pos in passage["entities"]]
            ents = [passage["text"][start:end+1] for start, end in ents_pos]
            for qa in data["qas"]:
                qa_id = f"{set_type}-{data['idx']}-{qa['idx']}"
                answers = set([ans["text"] for ans in qa["answers"]])
                for ent_idx, ent in enumerate(ents):
                    is_answer = ent in answers
                    guid = f"{qa_id}-{ent_idx}"
                    examples.append(
                        InputExample(
                            guid=guid,
                            text_a=passage["text"],
                            # Insert entity in query
                            text_b=qa["query"].replace("@placeholder", ent),
                            label="1" if is_answer else "0",
                        )
                    )
        return examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]


superglue_processors = {
    "boolq": BoolQProcessor,
    "cb": CBProcessor,
    "copa": COPAProcessor,
    "record": ReCoRDProcessor,
    "multirc": MultiRCProcessor,
}
