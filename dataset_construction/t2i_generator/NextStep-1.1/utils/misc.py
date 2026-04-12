import os
import numpy as np
import random

import torch


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed + rank)

class LargeInt(int):
    def __new__(cls, value):
        if isinstance(value, str):
            units = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
            last_char = value[-1].upper()
            if last_char in units:
                num = float(value[:-1]) * units[last_char]
                return super(LargeInt, cls).__new__(cls, int(num))
            else:
                return super(LargeInt, cls).__new__(cls, int(value))
        else:
            return super(LargeInt, cls).__new__(cls, value)

    def __str__(self):
        value = int(self)
        if abs(value) < 1000:
            return f"{value}"
        for unit in ["", "K", "M", "B", "T"]:
            if abs(value) < 1000:
                return f"{value:.1f}{unit}"
            value /= 1000
        return f"{value:.1f}P"  # P stands for Peta, or 10^15

    def __repr__(self):
        return f'"{self.__str__()}"'  # Ensure repr also returns the string with quotes

    def __json__(self):
        return f'"{self.__str__()}"'

    def __add__(self, other):
        if isinstance(other, int):
            return LargeInt(super().__add__(other))
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)  # This ensures commutativity