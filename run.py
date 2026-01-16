import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_classification import Exp_Classification
from exp.exp_zero_shot_forecasting import Exp_Zero_Shot_Forecast
from utils.print_args import print_args
import random
import numpy as np
from args import get_args

if __name__ == "__main__":
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args = get_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device("cuda:{}".format(args.gpu))
        print("Using GPU")
    else:
        if hasattr(torch.backends, "mps"):
            args.device = (
                torch.device("mps")
                if torch.backends.mps.is_available()
                else torch.device("cpu")
            )
        else:
            args.device = torch.device("cpu")
        print("Using cpu or mps")

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print_args(args)

    if args.task_name == "long_term_forecast":
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == "short_term_forecast":
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == "imputation":
        Exp = Exp_Imputation
    elif args.task_name == "anomaly_detection":
        Exp = Exp_Anomaly_Detection
    elif args.task_name == "classification":
        Exp = Exp_Classification
    elif args.task_name == "zero_shot_forecast":
        Exp = Exp_Zero_Shot_Forecast
    else:
        Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii,
            )

            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting)
            if args.gpu_type == "mps":
                torch.backends.mps.empty_cache()
            elif args.gpu_type == "cuda":
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        setting = "{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}".format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.expand,
            args.d_conv,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii,
        )

        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == "mps":
            torch.backends.mps.empty_cache()
        elif args.gpu_type == "cuda":
            torch.cuda.empty_cache()
