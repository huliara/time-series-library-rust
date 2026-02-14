from _args_mock import Args_mock
from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import torch


def dataloader_test():
    args = Args_mock()
    exp = Exp_Long_Term_Forecast(args)
    device = exp.device
    _, dataloader = data_provider(args, flag="test")
    x = []
    y = []
    x_mark = []
    y_mark = []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in dataloader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            x.append(batch_x)
            y.append(batch_y)
            x_mark.append(batch_x_mark)
            y_mark.append(batch_y_mark)

    all_x = torch.cat(x, dim=0)
    all_y = torch.cat(y, dim=0)
    all_x_mark = torch.cat(x_mark, dim=0)
    all_y_mark = torch.cat(y_mark, dim=0)
    print("all_x shape:", all_x.shape)
    return (
        all_x.flatten().tolist(),
        all_y.flatten().tolist(),
        all_x_mark.flatten().tolist(),
        all_y_mark.flatten().tolist(),
    )


if __name__ == "__main__":
    x, y, x_stamp, y_stamp = dataloader_test()
    print("x shape:", len(x))
    print("stamp shape:", len(x_stamp))
