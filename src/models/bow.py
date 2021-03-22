import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F

INFINITY = 100


class BoWClassifier(nn.Module):

    def __init__(
            self,
            n_layers,
            embed_dim,
            hidden_dim,
            n_classes,
            dic,
            dropout=0.0,
    ):
        super().__init__()
        # Hyper parameters
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dic = dic
        # Word embeddings
        self.embed = nn.Embedding(
            len(dic),
            embed_dim,
            padding_idx=dic.pad_idx
        )
        # classifier layer
        dims = [embed_dim] + [hidden_dim] * (n_layers - 1) + [n_classes]
        layers = []
        for layer, (di, dh) in enumerate(zip(dims, dims[1:])):
            # Dropout
            layers.append(nn.Dropout(dropout))
            # Affine transform
            layers.append(nn.Linear(di, dh))
            # Non linearity
            if layer < n_layers - 1:
                layers.append(nn.ReLU())
        # Classifier
        self.output = nn.Sequential(*layers)

    def encode(self, x, lengths=None):
        """Encode sentence into embed_dim sized vectors"""
        L, bsz = x.size()
        # Embed words
        x_embeds = self.embed(x)
        # Mask padding tokens
        padding_mask = x.eq(self.dic.pad_idx).unsqueeze(-1)
        out = x_embeds.masked_fill(padding_mask, 0)
        # Mean pool
        out = out.mean(dim=0)
        # Rescale for shorter sentences
        out = out * L / (L-padding_mask.float().sum(dim=0))
        return out

    def logits(self, x, lengths=None):
        """Logits"""
        h = self.encode(x, lengths=lengths)
        return self.output(h)

    def nll(self, x, y, lengths=None, ls=0.0):
        """Negative log-likelihood"""
        # Get log priobabilities
        logits = self.logits(x, lengths=lengths)
        log_p = F.log_softmax(logits, dim=-1)
        # Log likelihood
        ll = th.gather(log_p, 1, y.unsqueeze(-1)).squeeze(-1)
        # Interpolate with the uniform distribution
        if ls > 0.0:
            ll = (1 - ls) * ll + ls * log_p.mean(dim=1)
        return -ll


class BoNgramLM(nn.Module):
    """[summary]

    Args:
        n_layers ([type]): [description]
        embed_dim ([type]): [description]
        hidden_dim ([type]): [description]
        context_size ([type]): [description]
        vocab_size ([type]): [description]
        dropout (float, optional): [description]. Defaults to 0.0.

    Returns:
        [type]: [description]
    """

    def __init__(
        self,
        n_layers,
        embed_dim,
        hidden_dim,
        context_size,
        vocab_size,
        dropout=0.0,
    ):
        super().__init__()
        # Hyper parameters
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_size = context_size
        self.vocab_size = vocab_size
        self.hidden_size = vocab_size
        # Word embeddings
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # Initial embeddings (for the -1, ... -context_size tokens)
        if self.context_size > 1:
            self.context_vectors = nn.Embedding(self.context_size-1, embed_dim)
        # classifier layer
        input_dim = embed_dim*context_size
        dims = [input_dim] + [embed_dim] * n_layers
        layers = []
        for layer, (di, dh) in enumerate(zip(dims, dims[1:])):
            # Dropout
            layers.append(nn.Dropout(dropout))
            # Affine transform
            layers.append(nn.Linear(di, dh))
            # Non linearity
            if layer < n_layers - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        # Classifier
        self.output = nn.Linear(embed_dim, vocab_size)

    def init_weights(self):
        nn.init.normal_(self.embed.weight, 0, 1/np.sqrt(self.embed_dim))
        for layer in self.layers:
            nn.init.normal_(layer.weight, 0, 1/np.sqrt(layer.in_features))
            nn.init.zero_(layer.bias)
        nn.init.normal_(self.output.weight, 0, 1/np.sqrt(self.embed_dim))
        nn.init.zero_(self.output.bias)

    def forward(self, input_ids, *args):
        bsz, L = input_ids.size()
        # Embed words
        embeds = self.embed(input_ids)
        # Pad to size
        if self.context_size > 1:
            pad_ids = th.arange(self.context_size-1).view(1, -1).repeat(bsz, 1)
            pad_vectors = self.context_vectors(pad_ids.to(embeds.device))
            embeds = th.cat([pad_vectors, embeds], dim=1)
        # Concat
        embeds = th.cat(
            [embeds[:, i:i+L]
             for i in range(self.context_size)],
            dim=-1
        )
        # Feed into the FF
        h = self.layers(embeds.view(-1, self.context_size*self.embed_dim))
        # Logits
        logits = self.output(h).view(bsz, L, -1)
        return logits


class BOWGenerative(nn.Module):
    """BoW generative model

    Args:
        n_layers ([type]): [description]
        embed_dim ([type]): [description]
        hidden_dim ([type]): [description]
        vocab_size ([type]): [description]
        n_classes ([type]): [description]
        dropout (float, optional): [description]. Defaults to 0.0.
        tie_embeddings (bool, optional): [description]. Defaults to True.
        generative (bool, optional): [description]. Defaults to False.
    """

    def __init__(
        self,
        vocab_size,
        n_classes,
        generative=False,
    ):
        super(BOWGenerative, self).__init__()
        # Hyper parameters
        self.vocab_size = vocab_size
        self.n_classes = n_classes
        self.hidden_size = vocab_size
        self.generative = generative
        # label logits
        self.label_logit = nn.Embedding(1, n_classes)
        # word logits
        self.word_logit = nn.Embedding(n_classes, vocab_size)

    def forward(
        self,
        labels,
        x,
        attention_mask=None,
        token_type_ids=None,
        return_embeds=False,
    ):
        """Encode sentence into 2 x dh sized vectors"""
        # Shape
        bsz, L = x.size()
        # Select logit for each word at each position depending on the label
        word_logits = self.word_logit(labels.view(-1, 1).repeat(1, L-1))
        word_logits = word_logits.view(bsz, (L-1), -1)
        # Get label logits
        if self.generative:
            label_logits = self.label_logit(th.zeros_like(labels))
            label_logits = label_logits.view(bsz, -1).contiguous()
            logits = (word_logits, label_logits)
        else:
            logits = (word_logits,)
        # Return logits
        return logits
