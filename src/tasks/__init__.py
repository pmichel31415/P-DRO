"""
Code for handling

Task = data + loss function

In some cases the task also contains a linear layer to project onto
the # of classes (when different tasks have different labels)
"""
import numpy as np

from .permuted_mnist import pMNIST
from .split_cifar import SplitCIFAR100
from .split_miniimagenet import SplitMiniImageNet
from .split_cub import SplitCUB
from .omniglot import Omniglot
from .waterbird import Waterbird
from .celeba import CelebA
from .split_mnist import SplitMNIST
from .text_classification import (
    GlueTask,
    TCDTask,
    MultiNLI,
    AmazonMultiDomainTask,
    MultiNLISagawa,
    BiasedSST,
    TestTextClassificationTask,
)
from .hatespeech_tasks import FountaTask, DavidsonTask
# from .superglue import SuperGlueTask, WiC, COPA, ReCoRD
from .language_modeling import (
    LanguageModelingTask,
    CCLanguageModelingTask,
)
from .image_density_estimation import ImageDensityEstimationTask
from . import language_modeling
from .task_suites import suites, prepare_task_suite
from .task import Task

task_list = {}


def _make_lm_fn(fn):
    """Creates the `LM task` version of the function call
    if fn(*args, **kwargs) returns a classification task,
    _make_lm_fn(fn)(*args, **kwargs) returns the associated LM task
    """
    def lm_fn(*args, **kwargs):
        task, input_shape, _ = fn(*args, **kwargs)
        input_shape = task.tokenizer.vocab_size
        lm_task = LanguageModelingTask.from_text_task(
            task)
        return lm_task, input_shape, None
    return lm_fn


def _make_cc_lm_fn(fn, generative=False):
    """Creates the `LM task` version of the function call
    if fn(*args, **kwargs) returns a classification task,
    _make_lm_fn(fn)(*args, **kwargs) returns the associated class conditional
    LM task
    """
    def lm_fn(*args, **kwargs):
        task, input_shape, _ = fn(*args, **kwargs)
        input_shape = task.tokenizer.vocab_size
        lm_task = CCLanguageModelingTask.from_text_task(
            task,
            generative=generative,
        )
        return lm_task, input_shape, task.n_classes
    return lm_fn


def register_task(can_be_lm=False, can_be_class_conditional_lm=False):

    def register_task_fn(fn):
        name = fn.__name__
        if name in task_list:
            raise ValueError(f"Cannot register duplicate task ({name})")
        if not hasattr(fn, "__call__"):
            raise ValueError("A task should be a function")
        task_list[name] = fn
        # Also register the LM version
        if can_be_lm:
            task_list[f"{name}_LM"] = _make_lm_fn(fn)
        if can_be_class_conditional_lm:
            task_list[f"{name}_gen_LM"] = _make_cc_lm_fn(fn, generative=True)
            task_list[f"{name}_cc_LM"] = _make_cc_lm_fn(fn)
        return fn

    return register_task_fn


@register_task()
def MNIST(path="./datasets", model_name=None):
    """10 digits of MNIST split into 5 2-way classification tasks"""
    return SplitMNIST(path, valid_split=10000), (1, 28, 28), 10


@register_task()
def MNIST_density(path="./datasets", model_name=None):
    """10 digits of MNIST split into 5 2-way classification tasks"""
    original_task = SplitMNIST(path, valid_split=10000)
    task = ImageDensityEstimationTask.from_image_task(original_task)
    return task, (1, 28, 28), 256


@register_task()
def CIFAR100(path="./datasets", model_name=None):
    """100 classes of CIFAR100 split into 20 5-way classification tasks"""
    return SplitCIFAR100(path, valid_split=10000), (3, 32, 32), 100


@register_task()
def CIFAR100_density(path="./datasets", model_name=None):
    """10 digits of MNIST split into 5 2-way classification tasks"""
    original_task = SplitCIFAR100(path, valid_split=10000)
    task = ImageDensityEstimationTask.from_image_task(original_task)
    return task, (3, 32, 32), 256


@register_task()
def MiniImageNet84(path="./datasets", model_name=None):
    """100 classes of MiniImageNet split into 20 5-way classification tasks"""
    task = SplitMiniImageNet(path, in_memory=True, split=[6, 2, 2])
    return task, (3, 84, 84), 100


@register_task()
def MiniImageNet256(path="./datasets", model_name=None):
    """100 classes of MiniImageNet split into 20 5-way classification tasks"""
    task = SplitMiniImageNet(path, in_memory=False, split=[
                             6, 2, 2], img_size=256)
    return task, (3, 256, 256), 100


@register_task()
def Waterbird28(path="./datasets", model_name=None):
    """Waterbird dataset"""
    task = Waterbird(path, img_size=28)
    return task, (3, 28, 28), 2


@register_task()
def Waterbird84(path="./datasets", model_name=None):
    """Waterbird dataset"""
    task = Waterbird(path, img_size=84)
    return task, (3, 84, 84), 2


@register_task()
def Waterbird224(path="./datasets", model_name=None):
    """Waterbird dataset"""
    task = Waterbird(path, img_size=224)
    return task, (3, 224, 224), 2


@register_task()
def Waterbird256(path="./datasets", model_name=None):
    """Waterbird dataset"""
    task = Waterbird(path, img_size=256)
    return task, (3, 256, 256), 2


@register_task()
def CelebA28(path="./datasets", model_name=None):
    """CelebA dataset"""
    task = CelebA(path, img_size=28, in_memory=False)
    return task, (3, 28, 28), 2


@register_task()
def CelebA84(path="./datasets", model_name=None):
    """CelebA dataset"""
    task = CelebA(path, img_size=84, in_memory=False)
    return task, (3, 84, 84), 2


@register_task()
def CelebA224(path="./datasets", model_name=None):
    """CelebA dataset"""
    task = CelebA(path, img_size=224, in_memory=False)
    return task, (3, 224, 224), 2


@register_task()
def CelebA256(path="./datasets", model_name=None):
    """CelebA dataset"""
    task = CelebA(path, img_size=256, in_memory=False)
    return task, (3, 256, 256), 2


@register_task()
def WikiText2(path="./datasets", model_name=None):
    """100 classes of MiniImageNet split into 20 5-way classification tasks"""
    task = language_modeling.WikiText(path, "2", model_name=model_name)
    input_shape = task.tokenizer.vocab_size
    return task, input_shape, None


@register_task()
def WikiText103(path="./datasets", model_name=None):
    """100 classes of MiniImageNet split into 20 5-way classification tasks"""
    task = language_modeling.WikiText(path, "103", model_name=model_name)
    input_shape = task.tokenizer.vocab_size
    return task, input_shape, None


@register_task()
def amazon_multi_domain(path="./datasets", model_name=None):
    """100 classes of MiniImageNet split into 20 5-way classification tasks"""
    task = AmazonMultiDomainTask(path, model_name=model_name)
    input_shape = task.tokenizer.vocab_size
    return task, input_shape, None


@register_task()
def AMD_LM(path="./datasets", model_name=None):
    """100 classes of MiniImageNet split into 20 5-way classification tasks"""
    task = AmazonMultiDomainTask(path, model_name=model_name)
    input_shape = task.tokenizer.vocab_size
    lm_task = language_modeling.LanguageModelingTask.from_text_task(task)
    return lm_task, input_shape, None


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def MNLI(path="./datasets", model_name=None):
    """Multi NLI"""
    task = MultiNLI(path, model_name=model_name)
    input_shape = task.tokenizer.vocab_size
    return task, input_shape, None


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def MNLI_sagawa(path="./datasets", model_name=None):
    """MultiNLI version from Sagawa et al. ICLR 2020"""
    task = MultiNLISagawa(path, model_name=model_name)
    input_shape = task.tokenizer.vocab_size
    return task, input_shape, None


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def test_text_classification(path="./datasets", model_name=None):
    """100 classes of MiniImageNet split into 20 5-way classification tasks"""
    # Make the biased version of SST
    task = TestTextClassificationTask(model_name=model_name)
    return task, task.tokenizer.vocab_size, task.n_classes


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def Founta(path="./datasets", model_name=None):
    """Hatespeech classification from Founta et al."""
    task = FountaTask(path, model_name=model_name)
    input_shape = task.tokenizer.vocab_size
    return task, input_shape, None


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def Davidson(path="./datasets", model_name=None):
    """Hatespeech classification from Davidson et al."""
    task = DavidsonTask(path, model_name=model_name)
    input_shape = task.tokenizer.vocab_size
    return task, input_shape, None


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def SST(path="./datasets", model_name=None):
    """100 classes of MiniImageNet split into 20 5-way classification tasks"""
    task = GlueTask(path, task_name="SST-2", model_name=model_name)
    input_shape = task.tokenizer.vocab_size
    return task, input_shape, None


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def SST_imbalanced(path="./datasets", model_name=None):
    """100 classes of MiniImageNet split into 20 5-way classification tasks"""
    task = GlueTask(path, task_name="SST-2", model_name=model_name)
    # Subsample training data
    pos_idxs = [idx for idx, z in enumerate(task.train_data)
                if z.label.item() == 1]
    neg_idxs = [idx for idx, z in enumerate(task.train_data)
                if z.label.item() == 0]
    # Subsample negative indices
    rng = np.random.RandomState(6843135)
    # 10:1 ratio
    neg_idxs = rng.choice(neg_idxs, size=len(pos_idxs)//10)
    new_idxs = pos_idxs
    new_idxs.extend(neg_idxs)
    task.train_data.inplace_subset(new_idxs)
    # Debugging
    input_shape = task.tokenizer.vocab_size
    return task, input_shape, None


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def biased_SST(path="./datasets", model_name=None):
    """Biased version of SST"""
    # Make the biased version of SST
    task = BiasedSST(path, model_name=model_name)
    return task, task.tokenizer.vocab_size, task.n_classes


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def biased_SST_95(path="./datasets", model_name=None):
    """Biased version of SST"""
    # Make the biased version of SST
    task = BiasedSST(path, model_name=model_name, bias_percent=95)
    return task, task.tokenizer.vocab_size, task.n_classes


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def biased_SST_95_noisy(path="./datasets", model_name=None):
    """Biased version of SST"""
    # Make the biased version of SST
    task = BiasedSST(path, model_name=model_name,
                     bias_percent=95, label_noise=0.1)
    return task, task.tokenizer.vocab_size, task.n_classes


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def biased_SST_95_very_noisy(path="./datasets", model_name=None):
    """Biased version of SST"""
    # Make the biased version of SST
    task = BiasedSST(path, model_name=model_name,
                     bias_percent=95, label_noise=0.2)
    return task, task.tokenizer.vocab_size, task.n_classes


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def biased_SST_95_very_very_noisy(path="./datasets", model_name=None):
    """Biased version of SST"""
    # Make the biased version of SST
    task = BiasedSST(path, model_name=model_name,
                     bias_percent=95, label_noise=0.3)
    return task, task.tokenizer.vocab_size, task.n_classes


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def biased_SST_95_very_noisy_40(path="./datasets", model_name=None):
    """Biased version of SST"""
    # Make the biased version of SST
    task = BiasedSST(path, model_name=model_name,
                     bias_percent=95, label_noise=0.4)
    return task, task.tokenizer.vocab_size, task.n_classes


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def biased_SST_95_very_noisy_50(path="./datasets", model_name=None):
    """Biased version of SST"""
    # Make the biased version of SST
    task = BiasedSST(path, model_name=model_name,
                     bias_percent=95, label_noise=0.5)
    return task, task.tokenizer.vocab_size, task.n_classes


@register_task(can_be_lm=True, can_be_class_conditional_lm=True)
def biased_SST_99(path="./datasets", model_name=None):
    """100 classes of MiniImageNet split into 20 5-way classification tasks"""
    # Make the biased version of SST
    task = BiasedSST(path, model_name=model_name, bias_percent=99)
    return task, task.tokenizer.vocab_size, task.n_classes


# @register_task()
# def SST_imbalanced_LM(path="./datasets", model_name=None):
#     task, input_shape, _ = SST_imbalanced(path, model_name)
#     lm_task = language_modeling.LanguageModelingTask.from_text_task(task)
#     return lm_task, input_shape, None


def prepare_task(name, path="./datasets", model_name=None):
    """Returns a task by name

    Args:
        name (str): Name of the task. Availabe tasks are in `task_list`
        path (str, optional): Path to folder containing the dataset.
            Defaults to "./datasets".
        model_name (str, optional): Model name to determine the input format.
            Defaults to None.

    Returns:
        tuple: A tuple containing the task, the data input shape
            (eg. for images we need to know the # of channels)
            and the output_size (number of classes when applicable)
    """
    if name not in task_list:
        raise ValueError(f"Unknown task: {name}")
    else:
        return task_list[name](path, model_name)


__all__ = [
    "pMNIST",
    "SplitMNIST",
    "SplitCIFAR100",
    "SplitMiniImageNet",
    "SplitCUB",
    "Omniglot",
    "Task",
    # "SST",
    "GlueTask",
    "SuperGlueTask",
    "TCDTask",
    "LanguageModelingTask",
    "CCLanguageModelingTask",
    "suites",
    "prepare_task_suite"
]
