import os
import sys
import psutil
import random
import time
import math
from contextlib import contextmanager

import torch
import numpy as np



def seed_everything(seed: int = 42, workers: bool = False):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def trace(title: str):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta_mem = m1-m0
    sign = "+" if delta_mem >= 0 else "-"
    delta_mem = math.fabs(delta_mem)
    delta_time = time.time() - t0
    print(f"[{m1:.2f}GB({sign}{delta_mem:.1f}GB): {delta_time:.1f}sec] {title}", file=sys.stderr)


def nearest_valid_size(input_size: int, downsample_rate: int) -> int:
    """Find nearest x `>= input_size` satisfying `(x // downsample_rate) % 32 == 0`"""
    while (input_size // downsample_rate) % 32 != 0:
        input_size += 1
    assert (input_size // downsample_rate) % 32 == 0
    return input_size


def pad_if_needed(
        x: np.ndarray,
        max_len: int,
        pad_value: float = 0.0,
) -> np.ndarray:
    if len(x) == max_len: 
        return x
    num_pad = max_len - len(x)
    n_dim = len(x.shape)
    pad_widths = [(0, num_pad)] + [(0,0) for _ in range(n_dim - 1)]

    return np.pad(x, pad_width=pad_widths, mode="constant", constant_values=pad_value)

