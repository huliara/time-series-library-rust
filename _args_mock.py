class Args_mock:
    def __init__(self):
        self.task_name = "long_term_forecast"
        self.is_training = 0
        self.model_id = "test"
        self.model = "PatchTST"

        # data loader
        self.data = "ETTh1"
        self.root_path = "./data/ETT/"
        self.data_path = "ETTh1.csv"
        self.features = "S"
        self.target = "OT"
        self.freq = "h"
        self.checkpoints = "./checkpoints/"

        # forecasting task
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 96
        self.seasonal_patterns = "Monthly"
        self.inverse = False

        # inputation task
        self.mask_rate = 0.25

        # anomaly detection task
        self.anomaly_ratio = 0.25

        # model define
        self.expand = 2
        self.d_conv = 4
        self.top_k = 5
        self.num_kernels = 6
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.d_model = 512
        self.n_heads = 8
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.moving_avg = 25
        self.factor = 1
        self.distil = True
        self.dropout = 0.1
        self.embed = "timeF"
        self.activation = "gelu"
        self.channel_independence = 1
        self.decomp_method = "moving_avg"
        self.use_norm = 1
        self.down_sampling_layers = 0
        self.down_sampling_window = 1
        self.down_sampling_method = None
        self.seg_len = 96

        # optimization
        self.num_workers = 0
        self.itr = 1
        self.train_epochs = 10
        self.batch_size = 32
        self.patience = 3
        self.learning_rate = 0.0001
        self.des = "test"
        self.loss = "MSE"
        self.lradj = "type1"
        self.use_amp = False

        # GPU
        self.use_gpu = True
        self.gpu = 0
        self.gpu_type = "mps"
        self.use_multi_gpu = False
        self.devices = "0,1,2,3"

        # de-stationary projector params
        self.p_hidden_dims = [128, 128]
        self.p_hidden_layers = 2

        # metrics (dtw)
        self.use_dtw = False

        # Augmentation
        self.augmentation_ratio = 0
        self.seed = 2
        self.jitter = False
        self.scaling = False
        self.permutation = False
        self.randompermutation = False
        self.magwarp = False
        self.timewarp = False
        self.windowslice = False
        self.windowwarp = False
        self.rotation = False
        self.spawner = False
        self.dtwwarp = False
        self.shapedtwwarp = False
        self.wdba = False
        self.discdtw = False
        self.discsdtw = False
        self.extra_tag = ""

        # TimeXer
        self.patch_len = 16

        # GCN
        self.node_dim = 10
        self.gcn_depth = 2
        self.gcn_dropout = 0.3
        self.propalpha = 0.3
        self.conv_channel = 32
        self.skip_channel = 32

        self.individual = False

        # TimeFilter
        self.alpha = 0.1
        self.top_p = 0.5
        self.pos = 1
