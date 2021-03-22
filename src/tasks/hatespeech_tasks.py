#!/usr/bin/env python3
import os.path
from ..data.hatespeech import hatespeech_processors
from .text_classification import TextClassificationTask


class FountaTask(TextClassificationTask):
    """Toxicity classification task from Founta et al. 2017"""

    def __init__(
        self,
        path,
        max_seq_length=128,
        model_name="bert-base-uncased",
        tokenizer=None,
    ):
        # Call constructor
        super(FountaTask, self).__init__(
            os.path.join(path, "hatespeech", "founta"),
            "founta",
            hatespeech_processors["founta"](),
            max_seq_length=max_seq_length,
            model_name=model_name,
            tokenizer=tokenizer,
        )


class DavidsonTask(TextClassificationTask):
    """Toxicity classification task from Davidson et al. 2018"""

    def __init__(
        self,
        path,
        max_seq_length=128,
        model_name="bert-base-uncased",
        tokenizer=None,
    ):
        # Call constructor
        super(DavidsonTask, self).__init__(
            os.path.join(path, "hatespeech", "davidson"),
            "davidson",
            hatespeech_processors["davidson"](),
            max_seq_length=max_seq_length,
            model_name=model_name,
            tokenizer=tokenizer,
        )
