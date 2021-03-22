import os.path

import torch as th
import torch.nn.functional as F
import numpy as np

from ..data import scoring, superglue
from ..data.cached_dataset import load_and_cache_examples
from ..data.minibatch import TupleMiniBatch
from ..utils import xavier_initialize

from .text_classification import TextClassificationTask
scorers = {
    "WiC": scoring.Accuracy(),
}


class SuperGlueTask(TextClassificationTask):
    """
    SuperGLUE task adapted from Huggingface transformers

    For use with BERT and other big pretrained models
    """

    def __init__(
        self,
        path,
        task_name,
        max_seq_length=128,
        model_name="bert-base-uncased",
        tokenizer=None,
        n_valid=0,
    ):
        # Check GLUE task
        if task_name.lower() not in superglue.superglue_processors:
            raise ValueError(f"SuperGLUE task not found: {task_name.lower()}")
        self.n_valid = n_valid
        # Call constructor
        super(SuperGlueTask, self).__init__(
            os.path.join(path, "superglue", task_name),
            task_name,
            superglue.superglue_processors[task_name.lower()](),
            max_seq_length=max_seq_length,
            model_name=model_name,
            tokenizer=tokenizer,
        )
        # GLUE name
        self._name = f"superglue-{self.task_name}"

    def load_data(self):
        self._train_data = load_and_cache_examples(
            self.path,
            self.task_name,
            self.tokenizer._tokenizer,
            self.processor,
            self.max_seq_length,
            model_type=self.model_name,
            split="train",
            overwrite_cache=False
        )
        self._valid_data = load_and_cache_examples(
            self.path,
            self.task_name,
            self.tokenizer._tokenizer,
            self.processor,
            self.max_seq_length,
            model_type=self.model_name,
            split="dev",
            overwrite_cache=False
        )
        self._test_data = self._valid_data
        self._make_validation_set()
        print(len(self.train_data), len(self.valid_data), len(self.test_data))

    def _make_validation_set(self):
        # Build validation data out of the training set
        if self.n_valid > 0:
            # Fixed rng
            seed = (hash(self._name)+154) % (2**32)
            rng = np.random.RandomState(seed=seed)
            order = rng.permutation(len(self._train_data))
            valid_idxs = order[:self.n_valid]
            self._valid_data = self._train_data.subset(valid_idxs)
            train_idxs = order[self.n_valid:]
            self._train_data.inplace_subset(train_idxs)


class ReCoRD(SuperGlueTask):
    """Class for the ReCoRD"""

    def __init__(
        self,
        path,
        max_seq_length=128,
        model_type="bert-base-uncased",
    ):
        super(ReCoRD, self).__init__(
            path, "ReCoRD", max_seq_length, model_type)

    def eval_model(self, model, batch_size=32, data="test"):
        """Special evaluation function because we need to track the answers"""
        # Set model to test mode
        mode = model.training
        model.train(mode=False)
        # Select dataset for evaluation
        dataset = data
        if data == "test":
            dataset = self.test_data
            examples = self.processor.get_test_examples()
        elif data == "valid" or data == "dev":
            dataset = self.valid_data
            examples = self.processor.get_dev_examples()
        else:
            raise ValueError(
                "`data` must be a pytorch dataset or one of 'dev'/'valid'"
            )
        # Dataloader
        data_loader = th.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
        )
        y, y_hat = [], []
        for batch in data_loader:
            # Get model predictions
            with th.no_grad():
                _, predicted = self.predict(model, batch)
            # Track predictions and reference
            y.append(batch[-1])
            y_hat.append(predicted)

        y = th.cat(y, dim=0).cpu().numpy()
        y_hat = th.cat(y_hat, dim=0).cpu().numpy()

        # Aggregate predictions for each question
        by_question = {"y": {}, "y_hat": {}}
        for ex, y_i, y_i_hat in zip(examples, y, y_hat):
            qid = '-'.join(ex.example_id.split("-")[:-1])
            if qid not in by_question["y"]:
                by_question["y"][qid] = []
                by_question["y_hat"][qid] = []
            by_question["y"][qid].append(y_i)
            by_question["y_hat"][qid].append(y_i_hat)
        # Now Exact match over all questions
        compute_f1 = scoring.Accuracy()
        EMs = []
        for question in by_question.values():
            EMs.append(compute_f1(question["y_hat"], question["y"]))
        # Reset model to the original mode
        model.train(mode=mode)

        return sum(EMs) / len(EMs)


class MultiRC(SuperGlueTask):
    """Class for the MultiRC"""

    def __init__(
        self,
        path,
        max_seq_length=128,
        model_type="bert-base-uncased",
    ):
        super(MultiRC, self).__init__(
            path, "MultiRC", max_seq_length, model_type)

    def eval_model(self, model, batch_size=32, data="test"):
        """Special evaluation function because we need to track the answers"""
        # Set model to test mode
        mode = model.training
        model.train(mode=False)
        # Select dataset for evaluation
        dataset = data
        if data == "test":
            dataset = self.test_data
            examples = self.processor.get_test_examples()
        elif data == "valid" or data == "dev":
            dataset = self.valid_data
            examples = self.processor.get_dev_examples()
        else:
            raise ValueError(
                "`data` must be a pytorch dataset or one of 'dev'/'valid'"
            )
        # Dataloader
        data_loader = th.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=self.collate_fn,
        )
        y, y_hat = [], []
        for batch in data_loader:
            # Get model predictions
            with th.no_grad():
                _, predicted = self.predict(model, batch)
            # Track predictions and reference
            y.append(batch[-1])
            y_hat.append(predicted)

        y = th.cat(y, dim=0).cpu().numpy()
        y_hat = th.cat(y_hat, dim=0).cpu().numpy()

        # Aggregate predictions for each question
        by_question = {"y": {}, "y_hat": {}}
        for ex, y_i, y_i_hat in zip(examples, y, y_hat):
            qid = '-'.join(ex.example_id.split("-")[:-1])
            if qid not in by_question["y"]:
                by_question["y"][qid] = []
                by_question["y_hat"][qid] = []
            by_question["y"][qid].append(y_i)
            by_question["y_hat"][qid].append(y_i_hat)
        # Now F1 over all questions
        compute_f1 = scoring.F1()
        F1s = []
        for question in by_question.values():
            F1s.append(compute_f1(question["y_hat"], question["y"]))
        # Reset model to the original mode
        model.train(mode=mode)

        return sum(F1s) / len(F1s)


def average_word_embedding(h, word_pos):
    """Average embedding for a word

    Arguments:
        h {th.Tensor} -- B x L x D tensor of batched word embedding sequences
        word_pos {th.LongTensor} -- B x 2 tensor of word start and end index

    Returns:
        th.Tensor -- ``out[b] = h[b, word_pos[b, 0]:word_pos[b, 1]].mean()``
    """
    pos = th.arange(h.size(1)).to(h.device).view(1, -1, 1)
    start, end = word_pos[:, 0], word_pos[:, 1]
    word_mask = (pos >= start.view(-1, 1, 1)) & (pos < end.view(-1, 1, 1))
    word_mask = word_mask.float()
    h_word = (h*word_mask).sum(1) / word_mask.sum(1)
    return h_word


class WiC(SuperGlueTask):
    """Class for the WiC dataset"""

    def __init__(
        self,
        path,
        max_seq_length=128,
        model_type="bert-base-uncased",
    ):
        super(WiC, self).__init__(path, "WiC", max_seq_length, model_type)

    def nll(self, model, batch, reduction="mean"):
        """Compute the NLL loss for this task"""
        device = list(model.parameters())[0].device
        (
            input_ids,
            _,
            attention_mask,
            token_type_ids,
            labels,
        ) = batch.to(device)
        # Treat all choices as one big batch
        h = model(input_ids, attention_mask, token_type_ids)
        return self.nll_on_features(h, batch, reduction)

    def nll_on_features(self, h, batch, reduction="mean"):
        """Compute the NLL loss for this task"""
        _, words_pos, _, _, y = batch.to(h.device)
        # In WiC we also concatenate the words indices
        h_word_1 = average_word_embedding(h, words_pos[:, 0])
        h_word_2 = average_word_embedding(h, words_pos[:, 1])
        h_cls = h[:, 0]
        features = th.cat([h_word_1, h_word_2, h_cls], dim=1)
        # Log loss
        logits = self.head(features)
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, y, reduction=reduction)
        return nll_loss

    def predict(self, model, batch):
        """Predict label on this batch"""
        device = list(model.parameters())[0].device
        (
            input_ids,
            words_pos,
            attention_mask,
            token_type_ids,
            labels,
        ) = batch.to(device)
        # Extract features with th
        # Extract features with the model
        h = model(input_ids, attention_mask, token_type_ids)
        # In WiC we also concatenate the words indices
        h_word_1 = average_word_embedding(h, words_pos[:, 0])
        h_word_2 = average_word_embedding(h, words_pos[:, 1])
        h_cls = h[:, 0]
        features = th.cat([h_word_1, h_word_2, h_cls], dim=1)
        # predictions
        return self.predict_on_features(features)

    def build_head(self, n_features, device=None):
        """Build this tasks head"""
        # By default this is a linear layer
        self.head = th.nn.Linear(n_features*3, self.n_classes)
        xavier_initialize(self.head)
        if device is not None:
            self.head = self.head.to(device)


class MultipleChoiceTask(TextClassificationTask):

    def collate_fn(self, examples):
        n_choices = set([example[0].size(0) for example in examples])
        if len(n_choices) > 1:
            raise ValueError(
                f"Multiple choice tasks with variable choices are not"
                f"supported yet (got {n_choices})"
            )
        n_choices = n_choices.pop()
        bsz = len(examples)
        elements = [
            # input_ids
            th.stack([ex[0] for ex in examples]).view(bsz*n_choices, -1),
            # attention_mask
            th.stack([ex[1] for ex in examples]).view(bsz*n_choices, -1),
            # token_type_ids
            th.stack([ex[2] for ex in examples]).view(bsz*n_choices, -1),
            # Number of choices
            n_choices,
            # labels
            th.stack([ex[3] for ex in examples]),
        ]

        return TupleMiniBatch(elements)

    def build_head(self, n_features, device=None):
        """Build this tasks head"""
        # By default this is a linear layer
        self.head = th.nn.Linear(n_features, 1)
        xavier_initialize(self.head)
        if device is not None:
            self.head = self.head.to(device)

    def nll(self, model, batch, reduction="mean"):
        """Compute the NLL loss for this task"""
        device = list(model.parameters())[0].device
        (
            input_ids,
            attention_mask,
            token_type_ids,
            _,
            labels,
        ) = batch.to(device)
        # Treat all choices as one big batch
        h = model(input_ids, attention_mask, token_type_ids)
        return self.nll_on_features(h, batch, reduction)

    def nll_on_features(self, h, batch, reduction="mean"):
        """Compute the NLL loss given features h and targets y
        This assumes that the features have already be computed by the model"""
        _, _, _, num_choices, y = batch.to(h.device)
        num_choices = h.size(1)
        # Extract features with the model
        features = h.view(batch.size * num_choices, -1)
        # Log loss
        logits = self.head(features).view(batch.size, num_choices)
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = F.nll_loss(log_probs, y, reduction=reduction)
        return nll_loss

    def predict_on_features(self, h):
        """Predict label on this batch"""
        bsz = h.size(0)
        num_choices = h.size(1)
        features = h.view(bsz * num_choices, -1)
        logits = self.head(features).view(bsz, num_choices)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, logits.argmax(dim=-1)

    def predict(self, model, batch):
        """Predict label on this batch"""
        device = list(model.parameters())[0].device
        (
            input_ids,
            attention_mask,
            token_type_ids,
            num_choices,
            labels,
        ) = batch.to(device)
        # Treat all choices as one big batch
        h = model(input_ids, attention_mask, token_type_ids)
        # Reshape
        features = h[:, 0].view(batch.size, num_choices, -1)
        # predictions
        return self.predict_on_features(features)


class COPA(SuperGlueTask, MultipleChoiceTask):
    """Class for the WiC dataset"""

    def __init__(
        self,
        path,
        max_seq_length=128,
        model_type="bert-base-uncased",
    ):
        super(COPA, self).__init__(path, "COPA", max_seq_length, model_type)
