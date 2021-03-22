"""A pure python Embeddings implementation that supports double
differentiation"""
import torch as th
from torch import nn


class MyEmbedding(nn.Embedding):

    def forward(self, input):
        # Input has shape L x bsz
        embeds = th.stack([self.weight[x] for x in input])
        return embeds.masked_fill(input.eq(self.padding_idx).unsqueeze(-1), 0)

    def to_nn_embeddings(self):
        return nn.Embedding(
            self.num_embeddings,
            self.embedding_dim,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
            _weight=self.weight.data.clone(),
        )

    @staticmethod
    def from_nn_embeddings(nn_embeddings):
        return MyEmbedding(
            nn_embeddings.num_embeddings,
            nn_embeddings.embedding_dim,
            padding_idx=nn_embeddings.padding_idx,
            max_norm=nn_embeddings.max_norm,
            norm_type=nn_embeddings.norm_type,
            scale_grad_by_freq=nn_embeddings.scale_grad_by_freq,
            sparse=nn_embeddings.sparse,
            _weight=nn_embeddings.weight.data.clone(),
        )
