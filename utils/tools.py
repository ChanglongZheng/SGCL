import torch
import numpy as np
import random
import pickle
import os
import datetime


def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def seed_everything(seed, reproducibility=True):
    """
    init random seed for random functions in numpy, torch, cuda and cudnn
    :param seed: random seed
    :param reproducibility: whether to require reproducibility
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Sets the seed for PyTorch's random number generator on the CPU.
    torch.cuda.manual_seed(seed)  # Sets the seed for the random number generator on a specific GPU.
    torch.cuda.manual_seed_all(seed)  # Sets the seed for all GPUs if you are using multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现（看见有人用，我也加上）
    if reproducibility:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def set_color(log, color, highlight=True):
    # the ANSI CODE used for setting the color of the infos printed in the terminal, correspond code for colors are 30-37
    color_set = ['black', 'red', 'green', 'yellow', 'blue', 'pink', 'cyan', 'white']
    try:
        index = color_set.index(color)
    except ValueError:
        print('No color can be aligned with so the default color is setted.')
        index = 0
    log_prefix = '\033['
    log_affix = '\033[0m'
    if highlight:
        log_prefix += '1;3'
    else:
        log_prefix += '0;3'
    log_prefix += str(index) + 'm'
    return log_prefix + log + log_affix

