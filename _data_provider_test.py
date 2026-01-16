from data_provider.data_factory import data_provider
import numpy as np


def data_provider_test(args):
    args.augmentation_ratio = 0.0  # for test, do not use augmentation
    data_set, _ = data_provider(args, "test")
    x: np.ndarray = data_set.data_x
    y: np.ndarray = data_set.data_y
    return x, y
