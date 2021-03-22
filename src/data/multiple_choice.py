# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc.
# team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks
of reading comprehension

Slightly modified to suit my needs and assuage my linter"""


import csv
import glob
import json
import logging
import os
from typing import List

import tqdm

from transformers import (
    DataProcessor,
    InputExample,
    InputFeatures,
)


logger = logging.getLogger(__name__)


class MCInputExample(InputExample):
    """A single training/test example for multiple choice"""

    def __init__(self, guid, question, contexts, endings, label=None):
        """Constructs a MCInputExample.

        Args:
            guid: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence
                (context of corresponding question).
            question: string. The untokenized text of the second sequence
                (question).
            endings: list of str. multiple choice's options. Its length must
                be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label

    def to_features(
        self,
        tokenizer,
        max_length,
        label_map,
        pad_token_segment_id=0,
        pad_on_left=False,
        pad_token=0,
        mask_padding_with_zero=True,
    ):
        """Convert one example to input features"""
        choices_input_ids = []
        choices_attention_mask = []
        choices_token_type_ids = []

        for ending_idx, (context, ending) in enumerate(zip(
            self.contexts,
            self.endings
        )):
            text_a = context
            if self.question.find("@placeholder") != -1:
                # this is for cloze question
                text_b = self.question.replace("@placeholder", ending)
            else:
                text_b = self.question + " " + ending

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
            )
            if (
                "num_truncated_tokens" in inputs
                and inputs["num_truncated_tokens"] > 0
            ):
                print("WARNING: some tokens are getting truncated")

            input_ids = inputs["input_ids"]
            token_type_ids = inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens.
            # Only real tokens are attended to.
            attention_mask = [
                1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + attention_mask
                token_type_ids = (
                    [pad_token_segment_id] * padding_length
                ) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + \
                    ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + \
                    ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length

            choices_input_ids.append(input_ids)
            choices_attention_mask.append(attention_mask)
            choices_token_type_ids.append(token_type_ids)

        label = label_map[self.label]

        return InputFeatures(
            input_ids=choices_input_ids,
            attention_mask=choices_attention_mask,
            token_type_ids=choices_token_type_ids,
            label=label,
        )


# class MCInputFeatures(InputFeatures):
#     def __init__(self, guid, input_ids,input_mask,segment_ids, label):
#         self.guid = guid
#         self.input_ids = input_ids
#         self.input_mask = input_mask
#         self.segment_ids = segment_ids
#         self.label = label


class RaceProcessor(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        high = os.path.join(data_dir, "train/high")
        middle = os.path.join(data_dir, "train/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        high = os.path.join(data_dir, "dev/high")
        middle = os.path.join(data_dir, "dev/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        high = os.path.join(data_dir, "test/high")
        middle = os.path.join(data_dir, "test/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

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


class SwagProcessor(DataProcessor):
    """Processor for the SWAG data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        data = self._read_csv(os.path.join(data_dir, "train.csv"))
        return self._create_examples(data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        data = self._read_csv(os.path.join(data_dir, "val.csv"))
        return self._create_examples(data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        raise ValueError(
            "For swag testing, the input file does not contain a label column."
            " It can not be tested in current code setting!"
        )
        data = self._read_csv(os.path.join(data_dir, "test.csv"))
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_csv(self, input_file):
        with open(input_file, "r", encoding="utf-8") as f:
            return list(csv.reader(f))

    def _create_examples(self, lines: List[List[str]], type: str):
        """Creates examples for the training and dev sets."""
        if type == "train" and lines[0][-1] != "label":
            raise ValueError(
                "For training, the input file must contain a label column."
            )

        examples = [
            MCInputExample(
                guid=line[2],
                question=line[5],  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                contexts=[line[4], line[4], line[4], line[4]],
                endings=[line[7], line[8], line[9], line[10]],
                label=line[11],
            )
            for line in lines[1:]  # we skip the line with the column names
        ]

        return examples


class ArcProcessor(DataProcessor):
    """Processor for the ARC data set (request from allennlp)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        data = self._read_json(os.path.join(data_dir, "train.jsonl"))
        return self._create_examples(data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        data = self._read_json(os.path.join(data_dir, "dev.jsonl"))
        return self._create_examples(data, "dev")

    def get_test_examples(self, data_dir):
        data = self._read_json(os.path.join(data_dir, "test.jsonl"))
        return self._create_examples(data, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, type):
        """Creates examples for the training and dev sets."""

        # There are two types of labels. They should be normalized
        def normalize(truth):
            if truth in "ABCD":
                return ord(truth) - ord("A")
            elif truth in "1234":
                return int(truth) - 1
            else:
                logger.info("truth ERROR! %s", str(truth))
                return None

        examples = []
        three_choice = 0
        four_choice = 0
        five_choice = 0
        other_choices = 0
        # we deleted example which has more than or less than four choices
        for line in tqdm.tqdm(lines, desc="read arc data"):
            data_raw = json.loads(line.strip("\n"))
            if len(data_raw["question"]["choices"]) == 3:
                three_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) == 5:
                five_choice += 1
                continue
            elif len(data_raw["question"]["choices"]) != 4:
                other_choices += 1
                continue
            four_choice += 1
            truth = str(normalize(data_raw["answerKey"]))
            assert truth != "None"
            question_choices = data_raw["question"]
            question = question_choices["stem"]
            id = data_raw["id"]
            options = question_choices["choices"]
            if len(options) == 4:
                examples.append(
                    MCInputExample(
                        guid=id,
                        question=question,
                        contexts=[
                            options[0]["para"].replace("_", ""),
                            options[1]["para"].replace("_", ""),
                            options[2]["para"].replace("_", ""),
                            options[3]["para"].replace("_", ""),
                        ],
                        endings=[options[0]["text"], options[1]["text"],
                                 options[2]["text"], options[3]["text"]],
                        label=truth,
                    )
                )

        if type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None
        logger.info("len examples: %s}", str(len(examples)))
        logger.info("Three choices: %s", str(three_choice))
        logger.info("Five choices: %s", str(five_choice))
        logger.info("Other choices: %s", str(other_choices))
        logger.info("four choices: %s", str(four_choice))

        return examples


multichoice_processors = {
    "race": RaceProcessor,
    "swag": SwagProcessor,
    "arc": ArcProcessor,
}
