#!/usr/bin/env python3

import argparse


class DefaultValue(object):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value.__str__()


class ArgumentGroup(object):
    """This object groups arguments together.

    It extends the argument groups of argparse
    """

    def __init__(self, name, description=None):
        self.name = name
        self.description = description
        self.arguments = []
        self.is_default = {}
        self.in_name = {}

    def add_argument(
        self,
        *args,
        name=None,
        default=None,
        include_in_name=False,
        **kwargs,
    ):
        if name is None:
            name = args[0].strip("-").replace("-", "_")
        self.arguments.append(name)
        # Handle default value
        if "action" in kwargs:
            if kwargs["action"] == "store_true":
                default = False
            if kwargs["action"] == "store_false":
                default = True
        setattr(self, name, default)
        self.is_default[name] = True
        self.in_name[name] = include_in_name
        # These are the arguments that argparse will use
        argparse_kwargs = {k: v for k, v in kwargs.items()}
        # We wrap the default value around DefaultValue so we can determine
        # whether it was overwritten by command line arguments
        argparse_kwargs["default"] = DefaultValue(default)
        self.__setattr__(f"{name}_argparse", (args, argparse_kwargs))

    def add_to_parser(self, parser):
        group = parser.add_argument_group(self.name, self.description)
        for name in self.arguments:
            args, kwargs = getattr(self, f"{name}_argparse")
            group.add_argument(*args, **kwargs)

    def read_from_args(self, args):
        params_set = set(self.arguments)
        for k, v in args.__dict__.items():
            if k in params_set:
                # Check that the default was overwritten
                if not isinstance(v, DefaultValue):
                    self.__setattr__(k, v)
                    self.is_default[k] = False

    def __str__(self):
        lines = []
        lines.append("-"*100)
        lines.append(self.name)
        lines.append("-"*100)
        for name in sorted(self.arguments):
            lines.append(f"{name} : {getattr(self, name)}")
        return "\n".join(lines)


class Experiment(object):

    def __init__(self, name, description=None):
        self.name = name
        self.description = description
        self.configs = []
        self._configs_by_name = {}

    def add_configuration(self, config):
        self.configs.append(config)
        self._configs_by_name[config.name] = config

    def __getitem__(self, name):
        return self._configs_by_name[name]

    def make_exp_name(self):
        """Make an experiment name (for saving to files)

        Returns:
            str: a string describing the name
        """
        exp_name = []
        for config in self.configs:
            for param_name, in_name in config.in_name.items():
                # Either `in_name` is a bool, or it is determined by the value
                # of another parameter
                # This checks whether the parameter `in_name` is set to True
                if isinstance(in_name, str):
                    in_name = bool(getattr(config, in_name))
                # If the flag is False, then skip
                if not in_name:
                    continue
                # Get value of this parameter
                value = getattr(config, param_name)
                # Construct a string describing this parameter
                param_str = ""
                if isinstance(value, float):
                    # Handle floats precision
                    param_str = f"{param_name}_{value:.1e}"
                elif isinstance(value, bool):
                    # Handle flags
                    if not value:
                        continue
                    else:
                        param_str = param_name
                else:
                    # Otherwise default to [name]_[value]
                    param_str = f"{param_name}_{value}"
                # Add to the experiment name
                if param_name == "exp_prefix" and value is not None:
                    # the prefix is handled separately
                    exp_name.insert(0, value)
                else:
                    exp_name.append(param_str)
        # Join with "_"
        return "_".join(exp_name)

    def parse_args(self):
        self.parser = argparse.ArgumentParser(
            self.name,
            description=self.description,
        )
        # add configurations to parser
        for config in self.configs:
            config.add_to_parser(self.parser)
        # Parse
        args = self.parser.parse_args()
        # Add to configs
        for config in self.configs:
            config.read_from_args(args)
