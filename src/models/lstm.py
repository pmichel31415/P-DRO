import numpy as np
import torch as th
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

INFINITY = 100


class LSTMLM(nn.Module):
    """LSTM language model

    Args:
        n_layers ([type]): [description]
        embed_dim ([type]): [description]
        hidden_dim ([type]): [description]
        vocab_size ([type]): [description]
        dropout (float, optional): [description]. Defaults to 0.0.
        tie_embeddings (bool, optional): [description]. Defaults to True.
    """

    def __init__(
        self,
        n_layers,
        embed_dim,
        hidden_dim,
        vocab_size,
        dropout=0.0,
        tie_embeddings=True,
    ):
        super().__init__()
        # Hyper parameters
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.hidden_size = hidden_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        # Word embeddings
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=self.dropout,
            bidirectional=False,
        )
        # Initial states
        h0_init = th.randn(n_layers, 1, hidden_dim) / np.sqrt(hidden_dim)
        self.h0 = nn.Parameter(h0_init)
        c0_init = th.randn(n_layers, 1, hidden_dim) / np.sqrt(hidden_dim)
        self.c0 = nn.Parameter(c0_init)
        # Output layer to project onto the embedding space
        self.output = nn.Linear(hidden_dim, embed_dim)
        # Classifier
        self.last_drop = nn.Dropout(dropout)
        self.logit = nn.Linear(embed_dim, vocab_size)
        if tie_embeddings:
            self.logit.weight = self.embed.weight

    def train(self, training):
        # this is a hack so we can disable dropout by calling `.eval()` like in
        # other models. Otherwise if we call .eval on the nn.LSTM pytorch will
        # complain that it doesn't want to backprop through the cudnn lstm
        if training:
            self.lstm.dropout = self.dropout
            self.last_drop.train()
        else:
            self.lstm.dropout = 0
            self.last_drop.eval()

    def forward(
        self,
        x,
        attention_mask=None,
        token_type_ids=None,
        return_embeds=False,
    ):
        """Encode sentence into 2 x dh sized vectors"""
        # Shape
        bsz, L = x.size()
        # Embed words
        x_embeds = self.embed(x)
        # Hidden states
        hs = self.get_hidden_states(x_embeds, attention_mask, token_type_ids)
        # Project to embedding space with dropout
        hs = self.output(self.last_drop(hs.view(L*bsz, -1)))
        # Project to vocabulary
        logits = self.logit(hs).view(bsz, L, -1)
        # Return logits
        if return_embeds:
            return logits, x_embeds
        else:
            return logits

    def get_hidden_states(
        self,
        x_embeds,
        attention_mask=None,
        token_type_ids=None,
    ):
        bsz, L, _ = x_embeds.size()
        # Pack padded sequence if needed
        x_embeds = x_embeds.transpose(0, 1)
        if attention_mask is not None:
            lengths = attention_mask.float().sum(dim=-1).long()
            packed_embeds = pack_padded_sequence(x_embeds, lengths,
                                                 enforce_sorted=False)
        else:
            packed_embeds = x_embeds
        # Feed LSTM
        init_state = (self.h0.repeat(1, bsz, 1), self.c0.repeat(1, bsz, 1))
        out, _ = self.lstm(packed_embeds, init_state)
        if attention_mask is not None:
            out, _ = pad_packed_sequence(out, total_length=L)
        out = out.view(L, bsz, self.hidden_dim)
        return self.last_drop(out.transpose(0, 1)).contiguous()


class LSTMGenerative(LSTMLM):
    """LSTM generative model

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
        n_layers,
        embed_dim,
        hidden_dim,
        vocab_size,
        n_classes,
        dropout=0.0,
        tie_embeddings=True,
        generative=False,
    ):
        super(LSTMGenerative, self).__init__(
            n_layers, embed_dim,
            hidden_dim, vocab_size,
            dropout, tie_embeddings,
        )
        # Hyper parameters
        self.n_classes = n_classes
        self.generative = generative
        # label embeddings
        self.label_embed = nn.Embedding(n_classes, embed_dim)
        # Label softmax
        self.label_logit = nn.Linear(embed_dim, n_classes)
        if tie_embeddings:
            self.label_logit.weight = self.label_embed.weight

    def train(self, training):
        # same hack as the LM version
        if training:
            self.lstm.dropout = self.dropout
            self.last_drop.train()
        else:
            self.lstm.dropout = 0
            self.last_drop.eval()

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
        # Embed words (but not the EOS token)
        x_embeds = self.embed(x[:, :-1])
        # Embed labels
        y_embeds = self.label_embed(labels).view(bsz, 1, -1)
        # Input embeddings
        z_embeds = th.cat([y_embeds, x_embeds], dim=1)
        # Hidden states
        hs = self.get_hidden_states(z_embeds, attention_mask, token_type_ids)
        # Project and drop
        hs = self.output(self.last_drop(hs.view(L*bsz, -1))).view(bsz, L, -1)
        # Get h for words
        word_hs = hs[:, 1:].contiguous()
        # Word logits
        word_logits = self.logit(word_hs.view((L-1)*bsz, -1))
        word_logits = word_logits.view(bsz, (L-1), -1)
        # Get h for labels
        label_hs = hs[:, 1].contiguous()
        # Get label logits
        if self.generative:
            label_logits = self.label_logit(label_hs).view(bsz, -1)
            logits = (word_logits, label_logits)
        else:
            logits = (word_logits,)
        # Return logits
        if return_embeds:
            return logits, x_embeds
        else:
            return logits
