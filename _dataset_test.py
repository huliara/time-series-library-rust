from data_provider.data_loader import Dataset_ETT_hour
import numpy as np
from _args_mock import Args_mock


def dataset_test():
    args = Args_mock()
    dataset = Dataset_ETT_hour(
        args=args,
        root_path="./data/ETT/",
        flag="test",
        features="S",
        size=[args.seq_len, args.label_len, args.pred_len],
        data_path="ETTh1.csv",
        scale=True,
        timeenc=1,
        freq="h",
    )
    x: np.ndarray = dataset.data_x
    stamp: np.ndarray = dataset.data_stamp
    return x, stamp


if __name__ == "__main__":
    x, stamp = dataset_test()
    print("x shape:", x.shape)
    print("stamp shape:", stamp.shape)
