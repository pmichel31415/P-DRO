import torch as th
from transformers import GPT2PreTrainedModel, GPT2Model, GPT2Config
from torch import nn
from typing import Optional


class GPT2(GPT2PreTrainedModel):
    """A rewriting of GPT2LMHeadModel handling a single sentence at a time"""

    def __init__(self, config):
        super(GPT2, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.hidden_size = config.vocab_size
        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        past=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        transformer_outputs = self.transformer(
            input_ids,
            # past=past,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        return lm_logits


class GPT2ClassConditional(GPT2PreTrainedModel):
    """GPT2 but now it also takes labels as input"""

    def __init__(self, config, n_classes, generative=False):
        super(GPT2ClassConditional, self).__init__(config)
        self.n_classes = n_classes
        self.class_idxs = th.LongTensor([config.vocab_size + idx
                                         for idx in range(n_classes)])
        self.generative = generative
        # Create transformer model
        self.transformer = GPT2Model(config)
        # Add class embeddings
        self.transformer.resize_token_embeddings(
            config.vocab_size+self.n_classes
        )
        # Output layer
        self.lm_head = nn.Linear(
            config.n_embd,
            config.vocab_size+self.n_classes,
            bias=False,
        )

        self.hidden_size = config.vocab_size
        self.init_weights()

    def load_lm_state_dict(self, state_dict):
        """Initialize the LM parameters only from a LM model and don't touch
        the class-conditional parameters

        This is a bit more trickier than trivial because we share the
        embeddings for labels as well as words

        Args:
            state_dict: LM state dict
        """
        new_state_dict = {}
        for k, v in state_dict.items():
            if k == "model.transformer.wte.weight":
                w_embeds = v
                this_embeds = self.get_input_embeddings().weight
                if w_embeds.size(0) < this_embeds.size(0):
                    new_shape = (this_embeds.size(0), self.config.n_embd)
                    larger_w_embeds = w_embeds.new_empty(new_shape)
                    larger_w_embeds[:w_embeds.size(0)] = w_embeds
                    v = larger_w_embeds
            elif k == "model.lm_head.weight":
                weights = v
                this_weights = self.get_output_embeddings().weight
                if weights.size(0) < this_weights.size(0):
                    new_shape = (this_weights.size(0), self.config.n_embd)
                    larger_weights = weights.new_empty(new_shape)
                    larger_weights[:weights.size(0)] = weights
                    v = larger_weights
            elif k == "model.lm_head.bias":
                bias = v
                this_bias = self.get_output_embeddings().bias
                if bias.size(0) < this_bias.size(0):
                    new_shape = (this_bias.size(0),)
                    larger_bias = bias.new_empty(new_shape)
                    larger_bias[:bias.size(0)] = bias
                    v = larger_bias
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        self.load_state_dict(new_state_dict, strict=False)

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        labels,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        past=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        # prepend labels
        label_ids = self.class_idxs[labels].detach().to(input_ids.device)
        label_ids = label_ids.view(-1, 1)
        input_ids_with_label = th.cat(
            [input_ids[:, :1],      # Start of sentence token
             label_ids,             # label token
             input_ids[:, 1:-1]],   # Rest of the sentence(s) up until
            # (and excluding) the EOS token
            dim=1,
        )
        # Run the transformer
        transformer_outputs = self.transformer(
            input_ids_with_label,
            # past=past,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids,
            # position_ids=position_ids,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
        )
        # retrieve hidden states only
        hidden_states = transformer_outputs[0]
        if not self.generative:
            # Remove the hidden states associated with the labels
            # (we don't need to predict them)
            hidden_states = hidden_states[:, 1:].contiguous()
        # Compute logits
        lm_logits = self.lm_head(hidden_states)
        # Get word logits
        word_logits = lm_logits[:, 1:, :-self.n_classes].contiguous()
        # If this is a class conditional model
        if not self.generative:
            # Mask out/remove class logits
            # (we don't want to predict the classes)
            logits = (word_logits,)
        else:
            label_logits = lm_logits[:, 0, -self.n_classes:].contiguous()
            logits = (word_logits, label_logits)

        return logits


def small_transformer(
    n_layers: int,
    embed_dim: int,
    vocab_size: int,
    n_heads: int,
    n_classes: Optional[int] = None,
    generative: bool = False,
    from_lm: str = None,
):
    """Returns a small transformer with the same architecture as GPT-2 but
    different hyper-parameters.

    The name is a bit misleading since this can actually be as big as GPT-2 if
    we wanted...
    """
    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=embed_dim,
        n_layer=n_layers,
        n_head=n_heads,
    )
    if n_classes is None:
        # Unconditionla LM p(x)
        model = GPT2(config)
    elif not generative:
        # class conditional LM p(x|y)
        model = GPT2ClassConditional(config, n_classes)
    else:
        # Joint model p(x, y)
        model = GPT2ClassConditional(config, n_classes, generative=True)
    # Initialize from LM
    if from_lm is not None:
        lm_state_dict = th.load(from_lm)
        model.load_lm_state_dict(lm_state_dict)
    # Tie embeddings
    model.get_output_embeddings().weight = model.get_input_embeddings().weight
    return model
