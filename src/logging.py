import os.path
import numpy as np


class NpzLogger(object):
    """Logs experiment results to file

    Args:
        filename (str): Save file
        static_fields (dict, optional): These fields will be the same at
            every checkpoint (task name, stuff like that). Defaults to None.
        overwrite (bool, optional): Overwrite the log file if it already
            exists. Defaults to True.
    """

    def __init__(self, filename, static_fields=None, overwrite=True):
        self.filename = filename
        if os.path.isfile(filename):
            if overwrite:
                print(
                    f"Existing log file {filename} will be overwritten",
                    flush=True,
                )
            else:
                raise ValueError(f"Existing log file {filename} found")
        self.static_fields = {} if static_fields is None else static_fields
        self._appending = False

    def append(self, **kwargs):
        if self._appending:
            # Load previous logs
            prev_data = np.load(self.filename, allow_pickle=True)
            loaded_keys = prev_data.keys()
            keys_to_save = set(kwargs.keys()) | set(self.static_fields.keys())
            if len(loaded_keys - keys_to_save) > 0:
                raise ValueError(
                    f"Mismatch: keys in log file: {list(prev_data.keys())}"
                    f" and keys to save: {list(kwargs.keys())}"
                )
            prev_data = {k: prev_data[k].tolist() for k in prev_data}
        else:
            # Just initialize
            prev_data = {k: [] for k in kwargs}
            self._appending = True
        # Append data
        for key in kwargs:
            if key in self.static_fields:
                continue
            prev_data[key].append(kwargs[key])
        # Save
        np.savez_compressed(self.filename, **prev_data)
