from .mlp import MLP
from .smol_cnn import CNN
from .resnet import ResNetS
from .pretrained_resnet import make_headless_resnet18, make_headless_resnet50
from .bilstm import BiLSTMEncoder
from .lstm import LSTMLM, LSTMGenerative
from .bert import BERT, DistilBERT
from .gpt2 import GPT2, small_transformer
from .bow import BoNgramLM, BOWGenerative
import torch as th
from ..utils import xavier_initialize

from transformers import BertConfig

_PRETRAINED_MODELS_PATH = "pretrained_models"

architecture_list = {}


def register_architecture(alt_name=None):

    def register_architecture_fn(fn):
        name = fn.__name__
        if name in architecture_list:
            raise ValueError(
                f"Cannot register duplicate architecture ({name})")
        elif alt_name is not None and alt_name in architecture_list:
            raise ValueError(
                f"Cannot register duplicate architecture ({alt_name})")
        if not hasattr(fn, "__call__"):
            raise ValueError("A architecture should be a function")
        architecture_list[name] = fn
        if alt_name is not None:
            architecture_list[alt_name] = fn

        return fn

    return register_architecture_fn


class DummyClassifier(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 2

    def forward(self, x, *args, **kwargs):
        return th.zeros(len(x), self.hidden_size).to(x.device)


@register_architecture()
def dummy_classifier(input_shape, output_size):
    return DummyClassifier()


@register_architecture()
def mlp_2x400(input_shape, output_size):
    model = MLP(
        input_shape[0] * input_shape[1] * input_shape[2],
        output_size,
        hidden_size=400,
        hidden_layer_num=2,
        hidden_dropout_prob=.5,
        input_dropout_prob=.0,
    )
    # initialize the parameters.
    xavier_initialize(model)
    return model


@register_architecture()
def cnn_5(input_shape, output_size):
    return CNN(
        input_shape,
        output_size,
        kernels=[(3, 32), (3, 32), (3, 64), (3, 64), (3, 512)],
        hidden_dropout_prob=.5,
        input_dropout_prob=.0,
    )


@register_architecture()
def cnn_omniglot(input_shape, output_size):
    return CNN(
        input_shape,
        output_size,
        kernels=[(3, 64), (3, 64), (3, 64), (3, 64)],
        hidden_dropout_prob=.0,
        input_dropout_prob=.0,
        pool_every=1,
    )


@register_architecture()
def big_cnn_omniglot(input_shape, output_size):
    return CNN(
        input_shape,
        output_size,
        kernels=[(3, 128), (3, 128), (3, 128), (3, 128)],
        hidden_dropout_prob=.0,
        input_dropout_prob=.0,
        pool_every=1,
    )


@register_architecture()
def resnet_s(input_shape, output_size):
    return ResNetS(input_shape, num_classes=output_size)


@register_architecture()
def headless_resnet18(input_shape, output_size):
    return make_headless_resnet18(_PRETRAINED_MODELS_PATH)


@register_architecture()
def headless_resnet50(input_shape, output_size):
    return make_headless_resnet50(_PRETRAINED_MODELS_PATH)


@register_architecture()
def bilstm(input_shape, output_size):
    model = BiLSTMEncoder(
        n_layers=1,
        embed_dim=300,
        hidden_dim=300,
        vocab_size=input_shape,
        n_classes=output_size,
        dropout=0.1,
    )
    return model


@register_architecture()
def medium_bilstm(input_shape, output_size):
    model = BiLSTMEncoder(
        n_layers=2,
        embed_dim=300,
        hidden_dim=300,
        vocab_size=input_shape,
        n_classes=output_size,
        dropout=0.1,
    )
    return model


@register_architecture(alt_name="bert-base-uncased")
def bert_base_uncased(input_shape, output_size):
    return BERT.from_pretrained(
        "bert-base-uncased",
        cache_dir=_PRETRAINED_MODELS_PATH
    )


@register_architecture(alt_name="distilbert-base-uncased")
def distilbert_base_uncased(input_shape, output_size):
    return DistilBERT.from_pretrained(
        "distilbert-base-uncased",
        cache_dir=_PRETRAINED_MODELS_PATH
    )


@register_architecture(alt_name="bert-base-uncased-random")
def bert_base_uncased_random(input_shape, output_size):
    cfg = BertConfig.from_pretrained("bert-base-uncased")
    model = BERT(cfg)
    model.init_weights()
    return model


@register_architecture()
def gpt2(input_shape, output_size):
    return GPT2.from_pretrained(
        "gpt2",
        cache_dir=_PRETRAINED_MODELS_PATH
    )


@register_architecture()
def gpt2_untrained(input_shape, output_size):
    model = GPT2.from_pretrained(
        "gpt2",
        cache_dir=_PRETRAINED_MODELS_PATH
    )
    model.init_weights()
    return model


@register_architecture()
def small_lstm_lm(input_shape, output_size):
    return LSTMLM(1, 256, 512, input_shape, dropout=0.2, tie_embeddings=True)


@register_architecture()
def small_lstm_generative(input_shape, output_size):
    return LSTMGenerative(
        n_layers=1,
        embed_dim=256,
        hidden_dim=256,
        vocab_size=input_shape,
        n_classes=output_size,
        dropout=0.2,
        tie_embeddings=True,
        generative=True,
    )


@register_architecture()
def medium_lstm_lm(input_shape, output_size):
    return LSTMLM(2, 512, 512, input_shape, dropout=0.2, tie_embeddings=True)


@register_architecture()
def medium_lstm_generative(input_shape, output_size):
    return LSTMGenerative(
        n_layers=2,
        embed_dim=256,
        hidden_dim=512,
        vocab_size=input_shape,
        n_classes=output_size,
        dropout=0.2,
        tie_embeddings=True,
        generative=True,
    )


@register_architecture()
def bow_generative(input_shape, output_size):
    return BOWGenerative(
        vocab_size=input_shape,
        n_classes=output_size,
        generative=True,
    )


@register_architecture()
def small_transformer_lm(input_shape, output_size):
    return small_transformer(6, 512, input_shape, 8)


@register_architecture()
def small_transformer_class_conditional(input_shape, output_size):
    return small_transformer(
        6,
        512,
        input_shape,
        8,
        n_classes=output_size,
        generative=False,
    )


@register_architecture()
def small_transformer_generative(input_shape, output_size):
    return small_transformer(
        6,
        512,
        input_shape,
        8,
        n_classes=output_size,
        generative=True,
    )


@register_architecture()
def small_transformer_generative_wikitext103(input_shape, output_size):
    return small_transformer(
        6,
        512,
        input_shape,
        8,
        n_classes=output_size,
        generative=True,
        from_lm="pretrained_models/WikiText103_small_transformer_lm_model.pt",  # noqa
    )


@register_architecture()
def ff_lm(input_shape, output_size):
    return BoNgramLM(2, 256, 256, 10, input_shape, 0.1)


def build_model(
    name,
    input_shape,
    output_size=None,
    verbose=True
):
    """Build a model from a list of fixed architectures

    Args:
        name (str): Architecture name
        input_shape (object): An object describing the shape of the inputs
            (typically a tuple of integers for images or the vocabulary size
            for text models). This is usually taks and model dependent
        output_size (object, optional): [description]. An object describing
            the shape of the outputs. Typically this can be the number of
            classes, or the vocabulary size for text generating models. For
            "headless models" (models used as feature extractors for multiple
            tasks) this will be None. Defaults to None.
        verbose (bool, optional): Print out the model size (# params).
            Defaults to True.


    Returns:
        nn.Module: Pytorch module
    """

    if name not in architecture_list:
        raise ValueError(f"Unknown architecture: {name}")
    else:
        model = architecture_list[name](input_shape, output_size)

    num_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Number of model parameters: {num_params}")
    return model
