#!/usr/bin/env python3
"""Wrapper around pretrained resnets"""
from torch import nn
from torchvision import models
import os


def make_headless_resnet18(path):
    _torch_home = None
    if "TORCH_HOME" in os.environ:
        _torch_home = os.environ["TORCH_HOME"]
    os.environ["TORCH_HOME"] = os.path.abspath(path)
    model = models.resnet18(pretrained=True)
    if _torch_home is not None:
        os.environ["TORCH_HOME"] = _torch_home
    # set hidden size
    model.hidden_size = model.fc.in_features
    # Cut off head
    model.fc = nn.Identity()
    return model


def make_headless_resnet50(path):
    _torch_home = None
    if "TORCH_HOME" in os.environ:
        _torch_home = os.environ["TORCH_HOME"]
    os.environ["TORCH_HOME"] = os.path.abspath(path)
    model = models.resnet50(pretrained=True)
    if _torch_home is not None:
        os.environ["TORCH_HOME"] = _torch_home
    # set hidden size
    model.hidden_size = model.fc.in_features
    # Cut off head
    model.fc = nn.Identity()
    return model
