from transformers import BertPreTrainedModel, BertModel, DistilBertModel
import torch as th
from torch import nn


class BERT(BertPreTrainedModel):
    """BERT that only returns one vector"""

    def __init__(self, config):
        super(BERT, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # FIXME: no pooler so this is not equivalent to the standard BERT setup
        # Mercilessy ablate the pooler
        self.bert.pooler = nn.Identity()

        self.hidden_size = self.config.hidden_size

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # Reduce the size when possible?
        # TODO: this is now done at the dataset level already so it's
        # unnecessary here
        if attention_mask is not None:
            N = attention_mask.size(-1)
            longest = int(attention_mask.float().sum(-1).cpu().numpy().max())
            input_ids = input_ids[:, :longest]
            attention_mask = attention_mask[:, :longest]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, :longest]
            if position_ids is not None:
                position_ids = position_ids[:, :longest]
        # Run BERT
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # Get the last hidden states
        hs = outputs[0]
        # Pad again as needed
        if attention_mask is not None:
            padding_vecs = th.zeros(hs.size(0), N-longest, hs.size(-1))
            hs = th.cat([hs, padding_vecs.to(hs.device)], dim=1)
        assert hs.size(1) == N, "Bruh"
        # Dropout
        return self.dropout(hs)


class DistilBERT(BertPreTrainedModel):
    """DistilBERT that only returns one vector"""

    def __init__(self, config):
        super(DistilBERT, self).__init__(config)

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.hidden_size = self.config.hidden_size

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        # if token_type_ids is not None:
        #     print("Warning: token type id will go unused in distilbert")
        # if position_ids is not None:
        #     print("Warning: position id will go unused in distilbert")
        # if inputs_embeds is not None:
        #     print("Warning: inputs_embeds will go unused in distilbert")
        # Reduce the size when possible?
        # TODO: this is now done at the dataset level already so it's
        # unnecessary here
        if attention_mask is not None:
            N = attention_mask.size(-1)
            longest = int(attention_mask.float().sum(-1).cpu().numpy().max())
            input_ids = input_ids[:, :longest]
            attention_mask = attention_mask[:, :longest]
        # Run BERT
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
        )
        # Get the last hidden states
        hs = outputs[0]
        # Pad again as needed
        if attention_mask is not None:
            padding_vecs = th.zeros(hs.size(0), N-longest, hs.size(-1))
            hs = th.cat([hs, padding_vecs.to(hs.device)], dim=1)
        assert hs.size(1) == N, "Bruh"
        # Dropout
        return self.dropout(hs)
