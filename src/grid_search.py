
from copy import deepcopy


class Grid(object):

    def __init__(self, configs=None):
        self.configs = [GridSearchConfig()] if configs is None else configs

    def read_from_string(self, string, args=None):
        """Read grid configurations from a list of k=v1,...,v2
        args can be provided to check for unknown arguments"""
        for param in string:
            name, vals_str = param.split("=")
            vals = vals_str.split(",")
            # Check that the parameter exists in the namespace
            if args is not None and not hasattr(args, name):
                print(f"WARNING: ignoring unknown param {name} in grid search")
                continue
            # Update the grid
            new_configs = []
            for cfg in self.configs:
                for val in vals:
                    new_configs.append(cfg.add_to_copy(name, val))
            self.configs = new_configs

    def to_file(self, filename):
        with open(filename, "w") as fd:
            # Sort grid results in decreasing order
            grid_results = sorted(self.configs, key=lambda cfg: -cfg.score)
            for config in grid_results:
                print(f"{config}\t{config.score}", file=fd)

    def __getitem__(self, idx):
        return self.configs[idx]

    def __len__(self):
        return len(self.configs)


class GridSearchConfig(dict):

    def __init__(self, kwargs=None):
        self.kwargs = {} if kwargs is None else kwargs
        self.score = -1000000

    def add(self, key, value):
        self.kwargs[key] = value

    def add_to_copy(self, key, value):
        new_config = GridSearchConfig({k: v for k, v in self.kwargs.items()})
        new_config.add(key, value)
        return new_config

    def __str__(self):
        return ",".join(f"{name}={val}" for name, val in self.kwargs.items())

    def __getitem__(self, key):
        return self.kwargs[key]

    def __iter__(self):
        return self.kwargs.__iter__()

    def __next__(self):
        return self.kwargs.__next__()

    def overwrite_arguments(self, args):
        for param_name in self:
            # cast to the original argument's type (and pray for the best)
            original_type = type(getattr(args, param_name))
            setattr(args, param_name, original_type(self[param_name]))
        return args

    def arguments(self, args):
        return self.overwrite_arguments(deepcopy(args))
