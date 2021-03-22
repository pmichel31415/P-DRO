#!/usr/bin/env python3
"""Suites of tasks for multitask learning"""
import numpy as np

from .permuted_mnist import pMNIST
from .split_cifar import (
    SplitCIFAR100,
    coarse_to_fine_labels,
    fine_labels,
    CIFAR100,
)
from .split_miniimagenet import SplitMiniImageNet
from .omniglot import Omniglot
from .split_mnist import SplitMNIST
# from .sst import SST
from .text_classification import (
    GlueTask,
    TCDTask,
    MultiNLI,
    AmazonMultiDomainTask,
    MultiNLISagawa,
    BiasedSST,
)
from .superglue import SuperGlueTask, WiC, COPA, ReCoRD
from ..data import tokenizers

suites = {}


def register_task_suite():

    def register_task_suite_fn(fn):
        name = fn.__name__
        if name in suites:
            raise ValueError(f"Cannot register duplicate task suite ({name})")
        if not hasattr(fn, "__call__"):
            raise ValueError("A task suite should be a function")
        suites[name] = fn
        return fn

    return register_task_suite_fn


@register_task_suite()
def permuted_MNIST_20(path="./datasets", model_name=None):
    """Each task is a copy of MNIST with a different fixed
    permutation of the pixels"""
    rng = np.random.RandomState(seed=69)
    task_list = [pMNIST(path, valid_split=0, number=i, rng=rng)
                 for i in range(23)]
    input_shape = (1, 28, 28)
    output_size = 10
    return task_list, input_shape, output_size


@register_task_suite()
def split_MNIST(path="./datasets", model_name=None):
    """10 digits of MNIST split into 5 2-way classification tasks"""
    rng = np.random.RandomState(seed=69)
    digits_order = rng.permutation(10)
    task_list = [
        SplitMNIST(path, digits=set(digits_order[i:i+2]),
                   valid_split=0)
        for i in range(0, 10, 2)
    ]
    input_shape = (1, 28, 28)
    output_size = None
    return task_list, input_shape, output_size


@register_task_suite()
def split_CIFAR(path="./datasets", model_name=None):
    """100 classes of CIFAR100 split into 20 5-way classification tasks"""
    # Permutation chosen at random but fixed
    rng = np.random.RandomState(seed=69)
    classes_order = rng.permutation(100)
    print(classes_order)
    task_list = [
        SplitCIFAR100(
            "./datasets",
            classes=set(classes_order[i:i+5]),
            valid_split=0,
            cached_data=True,
        )
        for i in range(0, 100, 5)
    ]

    input_shape = (3, 32, 32)
    output_size = None
    return task_list, input_shape, output_size


@register_task_suite()
def split_CIFAR_traindev(path="./datasets", model_name=None):
    """100 classes of CIFAR100 split into 20 5-way classification tasks
    using the deterministic 40000/10000 train/dev split"""
    # Permutation chosen at random but fixed
    rng = np.random.RandomState(seed=69)
    classes_order = rng.permutation(100)
    print(classes_order)
    task_list = [
        CIFAR100(
            "./datasets",
            classes=set(classes_order[i:i+5]),
            valid_split=0,
            cached_data=True,
        )
        for i in range(0, 100, 5)
    ]

    input_shape = (3, 32, 32)
    output_size = None
    return task_list, input_shape, output_size


@register_task_suite()
def consistent_split_CIFAR(path="./datasets", model_name=None):
    """100 classes of CIFAR100 split into 20 5-way classification task
     according to coarse labels"""
    fine_label_idx = {label: idx for idx, label in enumerate(fine_labels)}
    task_list = [
        SplitCIFAR100(
            "./datasets",
            classes=set([fine_label_idx[label] for label
                         in coarse_to_fine_labels[coarse_label]]),
            valid_split=0,
            cached_train=True,
        )
        for coarse_label in coarse_to_fine_labels
    ]

    input_shape = (3, 32, 32)
    output_size = None
    return task_list, input_shape, output_size


@register_task_suite()
def split_MiniImageNet(path="./datasets", model_name=None):
    """100 classes of MiniImageNet split into 20 5-way classification tasks"""
    rng = np.random.RandomState(seed=69)
    classes_order = rng.permutation(100)
    task_list = [
        SplitMiniImageNet(
            "datasets",
            classes=set(classes_order[i:i+5]),
            in_memory=True,
            split=[8, 1, 1]
        )
        for i in range(0, 100, 5)
    ]
    input_shape = (3, 84, 84)
    output_size = None
    return task_list, input_shape, output_size


@register_task_suite()
def omniglot(path="./datasets", model_name=None):
    """50 character classification tasks from as many alphabets"""
    rng = np.random.RandomState(seed=69)
    alphabet_order = rng.permutation(50)
    task_list = [Omniglot(path, alphabet=a, in_memory=True)
                 for a in alphabet_order]
    input_shape = (1, 28, 28)
    output_size = None
    return task_list, input_shape, output_size


@register_task_suite()
def glue(path="./datasets", model_name="bert-base-uncased"):
    """GLUE: https://gluebenchmark.com/"""
    task_list = [
        GlueTask(path, task_name="SST-2", model_name=model_name),
        GlueTask(path, task_name="MNLI", model_name=model_name),
        GlueTask(path, task_name="MRPC", model_name=model_name),
        GlueTask(path, task_name="STS-B", model_name=model_name),
        GlueTask(path, task_name="QQP", model_name=model_name),
        GlueTask(path, task_name="CoLA", model_name=model_name),
        GlueTask(path, task_name="QNLI", model_name=model_name),
        GlueTask(path, task_name="RTE", model_name=model_name),
        GlueTask(path, task_name="WNLI", model_name=model_name),
    ]
    input_shape = None
    output_size = None
    return task_list, input_shape, output_size


@register_task_suite()
def tcd(path="./datasets", model_name="bert-base-uncased"):
    """Text classification datasets
    (same as https://papers.nips.cc/paper/9471-episodic-memory-in-lifelong-
    language-learning)"""
    task_list = [
        TCDTask(path, task_name="ag_news", model_name=model_name),
        TCDTask(path, task_name="amazon_review_full", model_name=model_name),
        # TCDTask(path, task_name="amazon_review_polarity",
        # model_name=model_name),
        TCDTask(path, task_name="dbpedia", model_name=model_name),
        TCDTask(path, task_name="yahoo_answers", model_name=model_name),
        TCDTask(path, task_name="yelp_review_full", model_name=model_name),
        # TCDTask(path, task_name="yelp_review_polarity",
        # model_name=model_name),
    ]
    input_shape = None
    output_size = None
    return task_list, input_shape, output_size


@register_task_suite()
def superglue(path="./datasets", model_name="bert-base-uncased"):
    """SuperGLUE: https://super.gluebenchmark.com/"""
    task_list = [
        SuperGlueTask(path, task_name="BoolQ", model_name=model_name),
        SuperGlueTask(path, task_name="CB", model_name=model_name),
        COPA(path, model_name=model_name),
        WiC(path, model_name=model_name),
        ReCoRD(path, model_name=model_name),
        SuperGlueTask(path, task_name="MultiRC", model_name=model_name),
    ]
    input_shape = None
    output_size = None
    return task_list, input_shape, output_size


@register_task_suite()
def mnli_by_genre(path="./datasets", model_name="bert-base-uncased"):
    """GLUE: https://gluebenchmark.com/"""
    tokenizer = tokenizers.PretrainedTokenizer("bert-base-uncased")
    genres = ["fiction", "government", "slate", "telephone", "travel",
              "facetoface", "letters", "nineeleven", "oup", "verbatim"]
    task_list = [
        MultiNLI(path, genres=[genre], tokenizer=tokenizer, cached_train=True,)
        for genre in genres
    ]
    input_shape = None
    output_size = None
    return task_list, input_shape, output_size


@register_task_suite()
def mnli_by_negation(path="./datasets", model_name="bert-base-uncased"):
    """Same setup as https://arxiv.org/abs/1911.08731"""
    tokenizer = tokenizers.PretrainedTokenizer(model_name)
    task_list = [
        MultiNLI(
            path,
            negations=negation,
            labels=[label],
            tokenizer=tokenizer,
            model_name=model_name,
            cached_train=True,
        )
        for negation in [True, False]
        for label in [0, 1, 2]
    ]
    return task_list, None, None


@register_task_suite()
def mnli_sagawa_by_negation(path="./datasets", model_name="bert-base-uncased"):
    """Same setup as https://arxiv.org/abs/1911.08731"""
    tokenizer = tokenizers.PretrainedTokenizer(model_name)
    task_list = [
        MultiNLISagawa(
            path,
            negations=negation,
            labels=[label],
            tokenizer=tokenizer,
            model_name=model_name,
            cached_train=True,
        )
        for negation in [True, False]
        for label in [0, 1, 2]
    ]
    return task_list, None, None


@register_task_suite()
def biased_sst(path="./datasets", model_name="bert-base-uncased"):
    """Same setup as https://arxiv.org/abs/1911.08731"""
    tokenizer = tokenizers.PretrainedTokenizer(model_name)
    task_list = [
        BiasedSST(
            path,
            tokenizer=tokenizer,
            model_name=model_name,
            label=label,
            biased=biased,
        )
        for label in range(2)
        for biased in [True, False]
    ]
    return task_list, tokenizer.vocab_size, 2


@register_task_suite()
def mnli_sagawa_by_genre(path="./datasets", model_name="bert-base-uncased"):
    """GLUE: https://gluebenchmark.com/"""
    tokenizer = tokenizers.PretrainedTokenizer("bert-base-uncased")
    genres = ["fiction", "government", "slate", "telephone", "travel",
              "facetoface", "letters", "nineeleven", "oup", "verbatim"]
    task_list = [
        MultiNLISagawa(
            path,
            genres=[genre],
            tokenizer=tokenizer,
            model_name=model_name,
            cached_train=True,
        )
        for genre in genres
    ]
    input_shape = None
    output_size = None
    return task_list, input_shape, output_size


@register_task_suite()
def amazon_multi_domain(path="./datasets", model_name="bert-base-uncased"):
    """GLUE: https://gluebenchmark.com/"""
    tokenizer = tokenizers.PretrainedTokenizer("bert-base-uncased")
    task_list = [AmazonMultiDomainTask(path, tokenizer=tokenizer)]
    return task_list, None, None


@register_task_suite()
def amazon_by_domain(path="./datasets", model_name="bert-base-uncased"):
    """GLUE: https://gluebenchmark.com/"""
    tokenizer = tokenizers.PretrainedTokenizer("bert-base-uncased")
    domains = [
        "apparel", "automotive", "baby", "books", "camera_&_photo",
        "cell_phones_&_service", "computer_&_video_games", "dvd",
        "electronics", "gourmet_food", "grocery", "health_&_personal_care",
        "jewelry_&_watches", "kitchen_&_housewares", "magazines", "music",
        "musical_instruments", "office_products", "outdoor_living", "software",
        "sports_&_outdoors", "tools_&_hardware", "toys_&_games", "video",
    ]
    task_list = [
        AmazonMultiDomainTask(
            path,
            tokenizer=tokenizer,
            domain=domain,
            cached_train=True,
        )
        for domain in domains
    ]
    return task_list, None, None


@register_task_suite()
def pan_nli(path="./datasets", model_name="bert-base-uncased"):
    """All NLI tasks"""
    task_list = [
        MultiNLI(path, genres=["fiction"], model_name=model_name),
        MultiNLI(path, genres=["government"], model_name=model_name),
        MultiNLI(path, genres=["slate"], model_name=model_name),
        MultiNLI(path, genres=["telephone"], model_name=model_name),
        MultiNLI(path, genres=["travel"], model_name=model_name),
        MultiNLI(path, genres=["facetoface"], model_name=model_name),
        MultiNLI(path, genres=["letters"], model_name=model_name),
        MultiNLI(path, genres=["nineeleven"], model_name=model_name),
        MultiNLI(path, genres=["oup"], model_name=model_name),
        MultiNLI(path, genres=["verbatim"], model_name=model_name),
        GlueTask(path, task_name="QNLI", model_name=model_name),
        GlueTask(path, task_name="WNLI", model_name=model_name),
    ]
    input_shape = None
    output_size = None
    return task_list, input_shape, output_size


@register_task_suite()
def pan_sentiment(path="./datasets", model_name="bert-base-uncased"):
    """All sentiment tasks"""
    task_list = [
        TCDTask(path, task_name="amazon_review_polarity",
                model_name=model_name),
        TCDTask(path, task_name="yelp_review_polarity", model_name=model_name),
        GlueTask(path, task_name="SST-2", model_name=model_name),
    ]
    input_shape = None
    output_size = None
    return task_list, input_shape, output_size


@register_task_suite()
def testing(path="./datasets", model_name="bert-base-uncased"):
    """SuperGLUE: https://super.gluebenchmark.com/"""
    task_list = [
        # SuperGlueTask(path, task_name="BoolQ", model_name=model_name),
        # SuperGlueTask(path, task_name="CB", model_name=model_name),
        # COPA(path),
        # WiC(path),
        # SuperGlueTask(path, task_name="MultiRC", model_name=model_name),
        ReCoRD(path),
    ]
    input_shape = None
    output_size = None
    return task_list, input_shape, output_size


@register_task_suite()
def text_classification(path="./datasets", model_name="bert-base-uncased"):
    """GLUE: https://gluebenchmark.com/"""
    tokenizer = tokenizers.PretrainedTokenizer(model_name)
    task_list = [
        TCDTask(path, task_name="ag_news",
                model_name=model_name, tokenizer=tokenizer),
        TCDTask(path, task_name="amazon_review_full",
                model_name=model_name, tokenizer=tokenizer),
        TCDTask(path, task_name="dbpedia",
                model_name=model_name, tokenizer=tokenizer),
        TCDTask(path, task_name="yahoo_answers",
                model_name=model_name, tokenizer=tokenizer),
        TCDTask(path, task_name="yelp_review_full",
                model_name=model_name, tokenizer=tokenizer),
        GlueTask(path, task_name="SST-2", n_valid=1000,
                 model_name=model_name, tokenizer=tokenizer),
        GlueTask(path, task_name="MNLI", n_valid=1000,
                 model_name=model_name, tokenizer=tokenizer),
        GlueTask(path, task_name="MRPC", n_valid=1000,
                 model_name=model_name, tokenizer=tokenizer),
        GlueTask(path, task_name="QQP", n_valid=1000,
                 model_name=model_name, tokenizer=tokenizer),
        GlueTask(path, task_name="CoLA", n_valid=1000,
                 model_name=model_name, tokenizer=tokenizer),
        GlueTask(path, task_name="QNLI", n_valid=1000,
                 model_name=model_name, tokenizer=tokenizer),
        GlueTask(path, task_name="RTE", n_valid=1000,
                 model_name=model_name, tokenizer=tokenizer),
        SuperGlueTask(path, task_name="BoolQ", n_valid=1000,
                      model_name=model_name, tokenizer=tokenizer),
        # WiC(path),
    ]
    # Arbitrary order
    rng = np.random.RandomState(seed=43092876)
    order = rng.permutation(len(task_list))
    task_list = [task_list[i] for i in order]
    input_shape = None
    output_size = None
    return task_list, input_shape, output_size


def prepare_task_suite(name, path="./datasets", model_name=None):
    if name not in suites:
        raise ValueError(f"Unknown task suite: {name}")
    else:
        return suites[name](path, model_name)
