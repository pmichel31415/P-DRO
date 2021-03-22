#!/usr/bin/env python3
"""Tokenizers for text data"""


from transformers import (
    BertTokenizer,
    OpenAIGPTTokenizer,
    GPT2Tokenizer,
    CTRLTokenizer,
    TransfoXLTokenizer,
    XLNetTokenizer,
    XLMTokenizer,
    DistilBertTokenizer,
    RobertaTokenizer,
    XLMRobertaTokenizer,
)


class Tokenizer(object):

    def __init__(self, vocab=None):
        self.vocab = vocab
        self.token_to_idx = {token: idx for idx, token in enumerate(vocab)}

    def tok(self, string):
        raise NotImplementedError()

    def __call__(self, string):
        return self.tok(string)

    def detok(self, list_of_strings):
        raise NotImplementedError()

    def encode_plus(
        self,
        string,
        text_pair=None,
        add_special_tokens=True,
        max_length=None,
    ):
        """Tokenize and convert to indices. Also return attention_masks"""
        output = {}
        if self.vocab is None:
            raise ValueError("No vocabulary provided")
        output["input_ids"] = [
            self.token_to_idx[token]
            for token in self.tok(string)
        ]
        output["token_type_ids"] = [0] * len(string)
        if text_pair is not None:
            output["input_ids"].extend([
                self.token_to_idx[token]
                for token in self.tok(text_pair)
            ])
            output["token_type_ids"].extend([1] * len(text_pair))
        if max_length is not None:
            output["input_ids"] = output["input_ids"][:max_length]
            output["token_type_ids"] = output["token_type_ids"][:max_length]
        return output

    def decode(self, idxs):
        """Convert indices to detokenized text"""
        if self.vocab is not None:
            return self.detok([self.vocab[idx] for idx in idxs])
        else:
            raise ValueError("No vocabulary provided")

    @property
    def vocab_size(self):
        return len(self.token_to_id)


class SpaceTokenizer(Tokenizer):
    """Tokenize along spaces"""

    def tok(self, string):
        return string.split(" ")

    def detok(self, list_of_strings):
        return " ".join(list_of_strings)


class CharTokenizer(Tokenizer):
    """Tokenize along spaces"""

    def tok(self, string):
        return list(string)

    def detok(self, list_of_strings):
        "".join(list_of_strings)


_PRETRAINED_TOKENIZERS = {
    "bert-base-uncased": BertTokenizer,
    "openai-gpt": OpenAIGPTTokenizer,
    "gpt2": GPT2Tokenizer,
    "ctrl": CTRLTokenizer,
    "transfo-xl-wt103": TransfoXLTokenizer,
    "xlnet-base-cased": XLNetTokenizer,
    "xlm-mlm-enfr-1024": XLMTokenizer,
    "distilbert-base-uncased": DistilBertTokenizer,
    "roberta-base": RobertaTokenizer,
    "xlm-roberta-base": XLMRobertaTokenizer,
}

_PRETRAINED_TOKENIZERS_PATH = "pretrained_models"


class PretrainedTokenizer(Tokenizer):
    """Pretrained subword tokenizer from Huggingface transformers
    https://github.com/huggingface/transformers
    """

    def __init__(self, model_name="bert-base-uncased"):
        tok_class = _PRETRAINED_TOKENIZERS[model_name]
        self._tokenizer = tok_class.from_pretrained(
            model_name,
            cache_dir=_PRETRAINED_TOKENIZERS_PATH,
        )

    def tok(self, string):
        return self._tokenizer.tokenize(string)

    def detok(self, list_of_strings):
        return self._tokenizer.convert_tokens_to_string(list_of_strings)

    def encode_plus(
        self,
        string,
        text_pair=None,
        add_special_tokens=True,
        max_length=None,
    ):
        # Defer to the tokenizer
        return self._tokenizer.encode_plus(
            string,
            text_pair=text_pair,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            truncation=max_length is not None,
        )

    @property
    def pad_token_id(self):
        if self._tokenizer.pad_token is not None:
            return self._tokenizer.pad_token_id
        else:
            return 0

    def decode(self, idxs, skip_special_tokens=False,
               clean_up_tokenization_spaces=True):
        """Convert indices to detokenized text"""
        return self._tokenizer.decode(idxs, skip_special_tokens,
                                      clean_up_tokenization_spaces)

    @property
    def vocab_size(self):
        return self._tokenizer.vocab_size
