from _args_mock import Args_mock
from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import torch


def dataset_test():
    args = Args_mock()
    exp = Exp_Long_Term_Forecast(args)
    device = exp.device
    _, dataloader = data_provider(args, flag="test")
    x = []
    y = []
    x_mark = []
    y_mark = []
    for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)
        x.append(batch_x)
        y.append(batch_y)
        x_mark.append(batch_x_mark)
        y_mark.append(batch_y_mark)

    all_x = torch.cat(all_x, dim=0)
    all_y = torch.cat(all_y, dim=0)
    all_x_mark = torch.cat(all_x_mark, dim=0)
    all_y_mark = torch.cat(all_y_mark, dim=0)
    return all_x, all_y, all_x_mark, all_y_mark


if __name__ == "__main__":
    x, stamp = dataset_test()
    print("x shape:", x.shape)
    print("stamp shape:", stamp.shape)
