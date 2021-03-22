import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

INFINITY = 100


class BiLSTMEncoder(nn.Module):
    """Simple BiLSTM"""

    def __init__(
        self,
        n_layers,
        embed_dim,
        hidden_dim,
        vocab_size,
        n_classes=None,
        dropout=0.0,
        pad_idx=0,
    ):
        super().__init__()
        # Hyper parameters
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.hidden_size = 2*hidden_dim
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        # Word embeddings
        self.embed = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=pad_idx
        )
        # LSTM
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
        )
        # Initial states
        h0_init = th.randn(2 * n_layers, 1, hidden_dim) / np.sqrt(hidden_dim)
        self.h0 = nn.Parameter(h0_init)
        c0_init = th.randn(2 * n_layers, 1, hidden_dim) / np.sqrt(hidden_dim)
        self.c0 = nn.Parameter(c0_init)
        # Classifier
        if n_classes is not None:
            self.output = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(2*hidden_dim, n_classes),
            )
        else:
            self.output = nn.Identity()

    def train(self, mode=True):
        """Sets the module in training mode.
        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in
        training/evaluation mode, if they are affected, e.g.
        :class:`Dropout`, :class:`BatchNorm`, etc.
        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.
        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            self.output.train(mode)
        return self

    def encode(self, x, lengths=None, return_embeds=False):
        """Encode sentence into 2 x dh sized vectors"""
        # Transpose x
        x = x.transpose(1, 0)
        # Shape
        L, bsz = x.size()
        # Embed words
        x_embeds = self.embed(x)
        # Pack padded sequence if needed
        if lengths is not None:
            packed_embeds = pack_padded_sequence(x_embeds, lengths,
                                                 enforce_sorted=False)
        else:
            packed_embeds = x_embeds
        # Feed LSTM
        init_state = (self.h0.repeat(1, bsz, 1), self.c0.repeat(1, bsz, 1))
        out, _ = self.bilstm(packed_embeds, init_state)
        if lengths is not None:
            out, _ = pad_packed_sequence(out)
        out = out.view(L, bsz, 2*self.hidden_dim)
        # out = out
        # Mask padding tokens
        padding_mask = x.eq(self.pad_idx).unsqueeze(-1)
        out = out.masked_fill(padding_mask, -INFINITY)
        # Max pool
        out = out.max(dim=0)[0]
        if return_embeds:
            return out, x_embeds
        else:
            return out

    def logits(self, x, lengths=None, return_embeds=False):
        """Logits"""
        h, x_embeds = self.encode(x, lengths=lengths, return_embeds=True)
        if return_embeds:
            return self.output(h), x_embeds
        else:
            return self.output(h)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
    ):
        """Return logits"""
        lengths = attention_mask.ne(0).long().sum(dim=-1)
        lengths = lengths.cpu().numpy().tolist()
        h = self.encode(input_ids, lengths)
        return h.unsqueeze(1)

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
