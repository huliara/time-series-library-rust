use crate::layers::embed::TokenEmbedding;
use burn::{
    config::Config,
    module::Module,
    nn::{
        pool::{AvgPool1d, AvgPool1dConfig},
        Dropout, DropoutConfig, Gelu, Linear, LinearConfig, PaddingConfig1d,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct TimeMixerConfig {
    pub task_name: String,
    pub seq_len: usize,
    pub label_len: usize,
    pub pred_len: usize,
    pub down_sampling_window: usize,
    pub channel_independence: bool,
    pub d_model: usize,
    pub d_ff: usize,
    pub dropout: f64,
    pub e_layers: usize,
    pub enc_in: usize,
    pub c_out: usize,
    #[config(default = "timeF")]
    pub embed: String,
    #[config(default = "h")]
    pub freq: String,
    pub use_norm: usize,
    pub down_sampling_layers: usize,
    pub down_sampling_method: String, // "max", "avg", "conv"
    pub decomp_method: String,        // "moving_avg", "dft_decomp"
    pub moving_avg: usize,
    pub top_k: usize,
    pub num_class: usize,
}

#[derive(Module, Debug)]
pub struct DataEmbeddingWoPos<B: Backend> {
    value_embedding: TokenEmbedding<B>,
    dropout: Dropout,
}

impl<B: Backend> DataEmbeddingWoPos<B> {
    pub fn new(c_in: usize, d_model: usize, dropout: f64, device: &B::Device) -> Self {
        let value_embedding = TokenEmbedding::new(c_in, d_model, device);
        let dropout = DropoutConfig::new(dropout).init();
        Self {
            value_embedding,
            dropout,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, _x_mark: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let x = self.value_embedding.forward(x);
        self.dropout.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct SeriesDecomp<B: Backend> {
    kernel_size: usize,
    avg: AvgPool1d<B>,
}

impl<B: Backend> SeriesDecomp<B> {
    pub fn new(kernel_size: usize, device: &B::Device) -> Self {
        let avg = AvgPool1dConfig::new(kernel_size)
            .with_stride(1)
            .with_padding(PaddingConfig1d::Explicit(0))
            .init(device);
        Self { kernel_size, avg }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let (batch_size, length, channel) = x.dims();
        let front = x
            .clone()
            .slice([0..batch_size, 0..1, 0..channel])
            .repeat_dim(1, (self.kernel_size - 1) / 2);
        let end = x
            .clone()
            .slice([0..batch_size, length - 1..length, 0..channel])
            .repeat_dim(1, (self.kernel_size - 1) / 2);

        let x_padded = Tensor::cat(vec![front, x.clone(), end], 1);

        // AvgPool1d expects [Batch, Channel, Length]
        let x_perm = x_padded.swap_dims(1, 2);
        let moving_mean = self.avg.forward(x_perm).swap_dims(1, 2);

        let res = x.sub(&moving_mean);
        (res, moving_mean)
    }
}

#[derive(Module, Debug)]
pub struct DFTSeriesDecomp<B: Backend> {
    top_k: usize,
    phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> DFTSeriesDecomp<B> {
    pub fn new(top_k: usize, _device: &B::Device) -> Self {
        Self {
            top_k,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        // TODO: Implement DFT.
        let trend = x.clone();
        let season = Tensor::zeros_like(&x);
        (season, trend)
    }
}

#[derive(Module, Debug)]
pub struct MultiscaleBlock<B: Backend> {
    down_linear_1: Linear<B>,
    down_linear_2: Linear<B>,
    activation: Gelu,
}

impl<B: Backend> MultiscaleBlock<B> {
    pub fn new(in_len: usize, out_len: usize, device: &B::Device) -> Self {
        Self {
            down_linear_1: LinearConfig::new(in_len, out_len).init(device),
            down_linear_2: LinearConfig::new(out_len, out_len).init(device),
            activation: Gelu::new(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.down_linear_1.forward(x);
        let x = self.activation.forward(x);
        self.down_linear_2.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct MultiScaleSeasonMixing<B: Backend> {
    down_sampling_layers: Vec<MultiscaleBlock<B>>,
}

impl<B: Backend> MultiScaleSeasonMixing<B> {
    pub fn new(config: &TimeMixerConfig, device: &B::Device) -> Self {
        let mut down_sampling_layers = Vec::new();
        for i in 0..config.down_sampling_layers {
            let in_len = config.seq_len / config.down_sampling_window.pow(i as u32);
            let out_len = config.seq_len / config.down_sampling_window.pow((i + 1) as u32);
            down_sampling_layers.push(MultiscaleBlock::new(in_len, out_len, device));
        }
        Self {
            down_sampling_layers,
        }
    }

    pub fn forward(&self, season_list: Vec<Tensor<B, 3>>) -> Vec<Tensor<B, 3>> {
        let mut out_high = season_list[0].clone();
        let mut out_low = season_list[1].clone();
        let mut out_season_list = Vec::new();

        out_season_list.push(out_high.clone());

        for i in 0..self.down_sampling_layers.len() {
            let out_low_res = self.down_sampling_layers[i].forward(out_high.clone());
            out_low = out_low + out_low_res;
            out_high = out_low.clone();

            if i + 2 < season_list.len() {
                out_low = season_list[i + 2].clone();
            }
            out_season_list.push(out_high.clone());
        }

        out_season_list
    }
}

#[derive(Module, Debug)]
pub struct MultiScaleTrendMixing<B: Backend> {
    up_sampling_layers: Vec<MultiscaleBlock<B>>,
}

impl<B: Backend> MultiScaleTrendMixing<B> {
    pub fn new(config: &TimeMixerConfig, device: &B::Device) -> Self {
        let mut up_sampling_layers = Vec::new();
        for i in (0..config.down_sampling_layers).rev() {
            let in_len = config.seq_len / config.down_sampling_window.pow((i + 1) as u32);
            let out_len = config.seq_len / config.down_sampling_window.pow(i as u32);
            up_sampling_layers.push(MultiscaleBlock::new(in_len, out_len, device));
        }
        Self { up_sampling_layers }
    }

    pub fn forward(&self, trend_list: Vec<Tensor<B, 3>>) -> Vec<Tensor<B, 3>> {
        let mut trend_list_reverse = trend_list.clone();
        trend_list_reverse.reverse();

        let mut out_low = trend_list_reverse[0].clone();
        let mut out_high = trend_list_reverse[1].clone();
        let mut out_trend_list = Vec::new();

        out_trend_list.push(out_low.clone());

        for i in 0..self.up_sampling_layers.len() {
            let out_high_res = self.up_sampling_layers[i].forward(out_low.clone());
            out_high = out_high + out_high_res;
            out_low = out_high.clone();

            if i + 2 < trend_list_reverse.len() {
                out_high = trend_list_reverse[i + 2].clone();
            }
            out_trend_list.push(out_low.clone());
        }

        out_trend_list.reverse();
        out_trend_list
    }
}

#[derive(Module, Debug)]
pub struct Normalize<B: Backend> {
    pub affine: bool,
    pub non_norm: bool,
    pub gamma: Option<Tensor<B, 1>>,
    pub beta: Option<Tensor<B, 1>>,
}

impl<B: Backend> Normalize<B> {
    pub fn new(num_features: usize, affine: bool, non_norm: bool, device: &B::Device) -> Self {
        let (gamma, beta) = if affine {
            (
                Some(Tensor::ones([num_features], device)),
                Some(Tensor::zeros([num_features], device)),
            )
        } else {
            (None, None)
        };
        Self {
            affine,
            non_norm,
            gamma,
            beta,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mode: &str) -> Tensor<B, 3> {
        if self.non_norm {
            return x;
        }
        match mode {
            "norm" => {
                let mean = x.clone().mean_dim(1);
                let var = x.clone().var(1);
                let std = (var + 1e-5).sqrt();
                let x = (x - mean) / std;
                if self.affine {
                    let g = self.gamma.as_ref().unwrap().clone().unsqueeze().unsqueeze();
                    let b = self.beta.as_ref().unwrap().clone().unsqueeze().unsqueeze();
                    x * g + b
                } else {
                    x
                }
            }
            "denorm" => x,
            _ => x,
        }
    }
}

#[derive(Module, Debug)]
pub struct PastDecomposableMixing<B: Backend> {
    seq_len: usize,
    pred_len: usize,
    down_sampling_window: usize,
    channel_independence: bool,

    decomp_method: String,
    series_decomp: Option<SeriesDecomp<B>>,
    dft_decomp: Option<DFTSeriesDecomp<B>>,

    cross_layer: Option<Linear<B>>,
    cross_layer_2: Option<Linear<B>>,
    cross_layer_activation: Gelu,

    mixing_multi_scale_season: MultiScaleSeasonMixing<B>,
    mixing_multi_scale_trend: MultiScaleTrendMixing<B>,

    out_cross_layer: Linear<B>,
    out_cross_layer_2: Linear<B>,
    out_cross_activation: Gelu,
}

impl<B: Backend> PastDecomposableMixing<B> {
    pub fn new(config: &TimeMixerConfig, device: &B::Device) -> Self {
        let series_decomp = if config.decomp_method == "moving_avg" {
            Some(SeriesDecomp::new(config.moving_avg, device))
        } else {
            None
        };

        let dft_decomp = if config.decomp_method == "dft_decomp" {
            Some(DFTSeriesDecomp::new(config.top_k, device))
        } else {
            None
        };

        let (cross_layer, cross_layer_2) = if !config.channel_independence {
            (
                Some(LinearConfig::new(config.d_model, config.d_ff).init(device)),
                Some(LinearConfig::new(config.d_ff, config.d_model).init(device)),
            )
        } else {
            (None, None)
        };

        let mixing_multi_scale_season = MultiScaleSeasonMixing::new(config, device);
        let mixing_multi_scale_trend = MultiScaleTrendMixing::new(config, device);

        let out_cross_layer = LinearConfig::new(config.d_model, config.d_ff).init(device);
        let out_cross_layer_2 = LinearConfig::new(config.d_ff, config.d_model).init(device);

        Self {
            seq_len: config.seq_len,
            pred_len: config.pred_len,
            down_sampling_window: config.down_sampling_window,
            channel_independence: config.channel_independence,
            decomp_method: config.decomp_method.clone(),
            series_decomp,
            dft_decomp,
            cross_layer,
            cross_layer_2,
            cross_layer_activation: Gelu::new(),
            mixing_multi_scale_season,
            mixing_multi_scale_trend,
            out_cross_layer,
            out_cross_layer_2,
            out_cross_activation: Gelu::new(),
        }
    }

    pub fn forward(&self, x_list: Vec<Tensor<B, 3>>) -> Vec<Tensor<B, 3>> {
        let mut length_list = Vec::new();
        for x in &x_list {
            length_list.push(x.dims()[1]);
        }

        let mut season_list = Vec::new();
        let mut trend_list = Vec::new();

        for x in &x_list {
            let (mut season, mut trend) = if self.decomp_method == "moving_avg" {
                self.series_decomp.as_ref().unwrap().forward(x.clone())
            } else {
                self.dft_decomp.as_ref().unwrap().forward(x.clone())
            };

            if !self.channel_independence {
                season = self.cross_layer.as_ref().unwrap().forward(season);
                season = self.cross_layer_activation.forward(season);
                season = self.cross_layer_2.as_ref().unwrap().forward(season);

                trend = self.cross_layer.as_ref().unwrap().forward(trend);
                trend = self.cross_layer_activation.forward(trend);
                trend = self.cross_layer_2.as_ref().unwrap().forward(trend);
            }

            season_list.push(season.swap_dims(1, 2));
            trend_list.push(trend.swap_dims(1, 2));
        }

        let out_season_list = self.mixing_multi_scale_season.forward(season_list);
        let out_trend_list = self.mixing_multi_scale_trend.forward(trend_list);

        let mut out_list = Vec::new();

        for i in 0..x_list.len() {
            let ori = &x_list[i];
            let length = length_list[i];
            let out_season = out_season_list[i].clone().swap_dims(1, 2);
            let out_trend = out_trend_list[i].clone().swap_dims(1, 2);

            let mut out = out_season.add(out_trend);

            if self.channel_independence {
                let out_res = self.out_cross_layer.forward(out.clone());
                let out_res = self.out_cross_activation.forward(out_res);
                let out_res = self.out_cross_layer_2.forward(out_res);
                out = ori.clone().add(out_res);
            }

            let slice = out.slice([0..out.dims()[0], 0..length, 0..out.dims()[2]]);
            out_list.push(slice);
        }

        out_list
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    configs: TimeMixerConfig,
    pdm_blocks: Vec<PastDecomposableMixing<B>>,
    preprocess: SeriesDecomp<B>,
    enc_embedding: DataEmbeddingWoPos<B>,
    normalize_layers: Vec<Normalize<B>>,
    predict_layers: Vec<Linear<B>>,
    projection_layer: Linear<B>,
    out_res_layers: Vec<Linear<B>>,
    regression_layers: Vec<Linear<B>>,
    classification_projection: Option<Linear<B>>,
}

impl<B: Backend> Model<B> {
    pub fn new(configs: TimeMixerConfig, device: &B::Device) -> Self {
        let mut pdm_blocks = Vec::new();
        for _ in 0..configs.e_layers {
            pdm_blocks.push(PastDecomposableMixing::new(&configs, device));
        }

        let preprocess = SeriesDecomp::new(configs.moving_avg, device);

        let enc_embedding = if configs.channel_independence {
            DataEmbeddingWoPos::new(1, configs.d_model, configs.dropout, device)
        } else {
            DataEmbeddingWoPos::new(configs.enc_in, configs.d_model, configs.dropout, device)
        };

        let mut normalize_layers = Vec::new();
        for _ in 0..=configs.down_sampling_layers {
            normalize_layers.push(Normalize::new(
                configs.enc_in,
                true,
                configs.use_norm == 0,
                device,
            ));
        }

        let mut predict_layers = Vec::new();
        let mut out_res_layers = Vec::new();
        let mut regression_layers = Vec::new();

        if configs.task_name == "long_term_forecast" || configs.task_name == "short_term_forecast" {
            for i in 0..=configs.down_sampling_layers {
                let in_len = configs.seq_len / configs.down_sampling_window.pow(i as u32);
                predict_layers.push(LinearConfig::new(in_len, configs.pred_len).init(device));

                if !configs.channel_independence {
                    out_res_layers.push(LinearConfig::new(in_len, in_len).init(device));
                    regression_layers
                        .push(LinearConfig::new(in_len, configs.pred_len).init(device));
                }
            }
        }

        let projection_layer = if configs.channel_independence {
            LinearConfig::new(configs.d_model, 1)
                .with_bias(true)
                .init(device)
        } else {
            LinearConfig::new(configs.d_model, configs.c_out)
                .with_bias(true)
                .init(device)
        };

        let classification_projection = if configs.task_name == "classification" {
            Some(
                LinearConfig::new(configs.d_model * configs.seq_len, configs.num_class)
                    .init(device),
            )
        } else {
            None
        };

        Self {
            configs,
            pdm_blocks,
            preprocess,
            enc_embedding,
            normalize_layers,
            predict_layers,
            projection_layer,
            out_res_layers,
            regression_layers,
            classification_projection,
        }
    }

    pub fn forward_forecast(
        &self,
        x_enc: Tensor<B, 3>,
        x_mark_enc: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let (x_enc_multi, x_mark_enc_multi) = self.multi_scale_process_inputs(x_enc, x_mark_enc);

        let mut x_list = Vec::new();
        let mut x_mark_list = Vec::new();

        for i in 0..x_enc_multi.len() {
            let x = x_enc_multi[i].clone();
            let mut x_norm = self.normalize_layers[i].forward(x, "norm");

            let (b, t, n) = x_norm.dims();
            if self.configs.channel_independence {
                x_norm = x_norm.swap_dims(1, 2).reshape([b * n, t, 1]);
                if let Some(ref marks) = x_mark_enc_multi {
                    let mark = marks[i].clone();
                    x_mark_list.push(Some(mark));
                } else {
                    x_mark_list.push(None);
                }
            } else {
                if let Some(ref marks) = x_mark_enc_multi {
                    x_mark_list.push(Some(marks[i].clone()));
                } else {
                    x_mark_list.push(None);
                }
            }
            x_list.push(x_norm);
        }

        let mut enc_out_list = Vec::new();
        let (out1_list, out2_list) = self.pre_enc(x_list.clone());

        for i in 0..out1_list.len() {
            let x_in = out1_list[i].clone();
            let mark = if i < x_mark_list.len() {
                x_mark_list[i].clone()
            } else {
                None
            };
            let enc_out = self.enc_embedding.forward(x_in, mark);
            enc_out_list.push(enc_out);
        }

        for block in &self.pdm_blocks {
            enc_out_list = block.forward(enc_out_list);
        }

        let dec_out_list =
            self.future_multi_mixing(out1_list.len(), enc_out_list, (out1_list, out2_list));

        let mut dec_out = dec_out_list[0].clone();
        for i in 1..dec_out_list.len() {
            dec_out = dec_out.add(dec_out_list[i].clone());
        }

        self.normalize_layers[0].forward(dec_out, "denorm")
    }

    fn pre_enc(&self, x_list: Vec<Tensor<B, 3>>) -> (Vec<Tensor<B, 3>>, Option<Vec<Tensor<B, 3>>>) {
        if self.configs.channel_independence {
            (x_list, None)
        } else {
            let mut out1_list = Vec::new();
            let mut out2_list = Vec::new();
            for x in x_list {
                let (x1, x2) = self.preprocess.forward(x);
                out1_list.push(x1);
                out2_list.push(x2);
            }
            (out1_list, Some(out2_list))
        }
    }

    fn future_multi_mixing(
        &self,
        count: usize,
        enc_out_list: Vec<Tensor<B, 3>>,
        x_lists: (Vec<Tensor<B, 3>>, Option<Vec<Tensor<B, 3>>>),
    ) -> Vec<Tensor<B, 3>> {
        let mut dec_out_list = Vec::new();
        let (_, x_list_res_opt) = x_lists;

        if self.configs.channel_independence {
            for i in 0..count {
                let enc_out = enc_out_list[i].clone();
                let enc_out_perm = enc_out.swap_dims(1, 2);
                let dec_out_perm = self.predict_layers[i].forward(enc_out_perm);
                let dec_out = dec_out_perm.swap_dims(1, 2);

                let dec_out_proj = self.projection_layer.forward(dec_out);
                dec_out_list.push(dec_out_proj);
            }
        } else {
            let x_list_res = x_list_res_opt.unwrap();
            for i in 0..count {
                let enc_out = enc_out_list[i].clone();
                let out_res = x_list_res[i].clone();

                let enc_out_perm = enc_out.swap_dims(1, 2);
                let dec_out_perm = self.predict_layers[i].forward(enc_out_perm);
                let mut dec_out = dec_out_perm.swap_dims(1, 2);

                dec_out = self.projection_layer.forward(dec_out);

                let out_res_perm = out_res.swap_dims(1, 2);
                let out_res_processed = self.out_res_layers[i].forward(out_res_perm);
                let out_res_regressed = self.regression_layers[i].forward(out_res_processed);
                let out_res_final = out_res_regressed.swap_dims(1, 2);

                dec_out = dec_out.add(out_res_final);
                dec_out_list.push(dec_out);
            }
        }
        dec_out_list
    }

    fn multi_scale_process_inputs(
        &self,
        x_enc: Tensor<B, 3>,
        x_mark_enc: Option<Tensor<B, 3>>,
    ) -> (Vec<Tensor<B, 3>>, Option<Vec<Tensor<B, 3>>>) {
        let mut x_enc_list = Vec::new();
        let mut x_mark_list = Vec::new();

        x_enc_list.push(x_enc.clone());
        if let Some(ref m) = x_mark_enc {
            x_mark_list.push(m.clone());
        }

        let mut current_x = x_enc;
        let mut current_mark = x_mark_enc;
        let device = &x_enc_list[0].device();

        for _ in 0..self.configs.down_sampling_layers {
            let x_perm = current_x.swap_dims(1, 2);
            let pool = AvgPool1dConfig::new(self.configs.down_sampling_window).init(device);
            let x_pooled = pool.forward(x_perm);
            current_x = x_pooled.swap_dims(1, 2);
            x_enc_list.push(current_x.clone());

            if let Some(ref m) = current_mark {
                x_mark_list.push(m.clone());
            }
        }

        let marks = if x_mark_list.is_empty() {
            None
        } else {
            Some(x_mark_list)
        };
        (x_enc_list, marks)
    }
}
