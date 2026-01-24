from data_provider.data_loader import Dataset_ETT_hour
import numpy as np
import argparse


def dataset_test():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.augmentation_ratio = 0.0  # for test, do not use augmentation
    dataset = Dataset_ETT_hour(
        args=args,
        root_path="./data/ETT/",
        flag="train",
        features="S",
        data_path="ETTh1.csv",
        scale=True,
        timeenc=0,
        freq="h",
    )
    x: np.ndarray = dataset.data_x
    stamp: np.ndarray = dataset.data_stamp
    return x, stamp


if __name__ == "__main__":
    x, stamp = dataset_test()
    print("x shape:", x.shape)
    print("stamp shape:", stamp.shape)
