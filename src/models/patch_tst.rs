use super::traits::Forecast;
use crate::args::TaskName;
use crate::layers::{
    embed::patch_embedding::PatchEmbedding,
    self_attention_family::{AttentionLayer, FullAttention},
    transformer_enc_dec::{Encoder, EncoderLayer},
};
use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use serde::{Deserialize, Serialize};

use clap::Args;
#[derive(Debug, Clone, Deserialize, Serialize, Args)]
pub struct PatchTSTArgs {
    #[arg(long)]
    pub seq_len: usize,
    #[arg(long)]
    pub pred_len: usize,
    #[arg(long)]
    pub num_class: usize,
    #[arg(long)]
    pub d_model: usize,
    #[arg(long, default_value_t = 16)]
    pub patch_len: usize,
    #[arg(long, default_value_t = 8)]
    pub stride: usize,
    #[arg(long)]
    pub enc_in: usize,
    #[arg(long)]
    pub e_layers: usize,
    #[arg(long)]
    pub n_heads: usize,
    #[arg(long)]
    pub d_ff: usize,
    #[arg(long)]
    pub dropout: f64,
    #[arg(long)]
    pub factor: usize,
    #[arg(long)]
    pub activation: String,
}

#[derive(Config, Debug)]
pub struct PatchTSTConfig {
    model_args: PatchTSTArgs,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl PatchTSTConfig {
    pub fn init<B: Backend>(&self, task_name: TaskName, device: &B::Device) -> PatchTST<B> {
        let configs = &self.model_args;
        let padding = configs.stride;
        let patch_embedding = PatchEmbedding::new(
            configs.d_model,
            configs.patch_len,
            configs.stride,
            padding,
            configs.dropout,
            device,
        );

        // Encoder
        // Create Encoder Layers
        let mut layers = Vec::new();
        for _ in 0..configs.e_layers {
            let attn_layer = AttentionLayer::new(
                FullAttention::new(false, configs.factor, None, configs.dropout, false),
                configs.d_model,
                configs.n_heads,
                None,
                None,
                device,
            );

            let layer = EncoderLayer::new(
                attn_layer,
                configs.d_model,
                Some(configs.d_ff),
                configs.dropout,
                configs.activation.clone(),
                device,
            );
            layers.push(layer);
        }

        let encoder = Encoder::new(layers, None);

        // Prediction Head
        let head_nf =
            configs.d_model * ((configs.seq_len - configs.patch_len) / configs.stride + 2);

        let head = if task_name == TaskName::LongTermForecast
            || task_name == TaskName::ShortTermForecast
        {
            Some(FlattenHead::new(
                head_nf,
                configs.pred_len,
                configs.dropout,
                device,
            ))
        } else if task_name == TaskName::Imputation || task_name == TaskName::AnomalyDetection {
            Some(FlattenHead::new(
                head_nf,
                configs.seq_len,
                configs.dropout,
                device,
            ))
        } else {
            None
        };

        let classification_projection = if task_name == TaskName::Classification {
            Some(LinearConfig::new(head_nf * configs.enc_in, configs.num_class).init(device))
        } else {
            None
        };

        PatchTST {
            patch_embedding,
            encoder,
            head,
            classification_projection,
            seq_len: configs.seq_len,
            pred_len: configs.pred_len,
            num_class: configs.num_class,
            d_model: configs.d_model,
            patch_len: configs.patch_len,
            stride: configs.stride,
            enc_in: configs.enc_in,
        }
    }
}

#[derive(Module, Debug)]
pub struct FlattenHead<B: Backend> {
    linear: Linear<B>,
    dropout: Dropout,
    nf: usize,
}

impl<B: Backend> FlattenHead<B> {
    pub fn new(nf: usize, target_window: usize, head_dropout: f64, device: &B::Device) -> Self {
        let linear = LinearConfig::new(nf, target_window).init(device);
        let dropout = DropoutConfig::new(head_dropout).init();

        Self {
            linear,
            dropout,
            nf,
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        // x: [bs, nvars, d_model, patch_num]
        // Flatten start_dim=-2 (last two dims: d_model, patch_num)
        // Check dimension order: Burn usually [Batch, ...].
        // Argument x passed here comes from Model which permuted it.
        // Model logic: enc_out = enc_out.permute(0, 1, 3, 2) -> [bs, nvars, d_model, patch_num]

        // Burn flatten:

        let x_flat = x.flatten(3, 4); // [bs, nvars, d_model * patch_num]
        let x_out = self.linear.forward(x_flat);
        self.dropout.forward(x_out)
    }
}

#[derive(Module, Debug)]
pub struct PatchTST<B: Backend> {
    patch_embedding: PatchEmbedding<B>,
    encoder: Encoder<B>, // Burn's Encoder

    // Heads
    head: Option<FlattenHead<B>>,
    classification_projection: Option<Linear<B>>,

    // Configs
    seq_len: usize,
    pred_len: usize,
    num_class: usize,
    d_model: usize,
    patch_len: usize,
    stride: usize,
    enc_in: usize,
}

impl<B: Backend> Forecast<B> for PatchTST<B> {
    fn forecast(
        &self,
        x_enc: Tensor<B, 3>,
        x_mark_enc: Tensor<B, 3>,
        _x_dec: Tensor<B, 3>,
        _x_mark_dec: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let means = x_enc.clone().mean_dim(1); // [Batch, 1, NVars]
        let x_enc = x_enc.sub(means.clone()); // Broadcast on dim 1

        let var = x_enc.clone().mul(x_enc.clone()).mean_dim(1); // Unbiased=False in torch means simple mean of squares of centered
        let stdev = (var + 1e-5).sqrt(); // [Batch, 1, NVars]
        let x_enc = x_enc.div(stdev.clone());

        // Patching & Embedding
        // x_enc: [Batch, Length, NVars] -> [Batch, NVars, Length] -> permute to handle channel independence
        // Logic: merge Batch and NVars.
        let dims = x_enc.dims();
        let (bs, seq_len, n_vars) = (dims[0], dims[1], dims[2]);
        let x_enc = x_enc.swap_dims(1, 2); // [B, N, L]
        let x_enc = x_enc.reshape([bs * n_vars, seq_len, 1]); // [B*N, L, 1]

        // Patch embedding
        // enc_out: [B*N, PatchNum, DModel]
        let (enc_out, _patch_num) = self.patch_embedding.forward(x_enc.clone());

        // Encoder
        // enc_out: [B*N, P, D]
        let (enc_out, _attns) = self.encoder.forward(enc_out, None);

        // Reshape back
        // enc_out: [B*N, P, D] -> [B, N, P, D]
        let patch_num = enc_out.dims()[1];
        let enc_out = enc_out.reshape([bs, n_vars, patch_num, self.d_model]);

        // [B, N, D, P] (for FlattenHead)
        let enc_out = enc_out.permute([0, 1, 3, 2]);

        // Decoder (Head)
        let dec_out = self.head.as_ref().unwrap().forward(enc_out); // [B, N, TargetWindow]
        let dec_out = dec_out.swap_dims(1, 2); // [B, Target, N]

        // De-Normalization
        // stdev: [Batch, 1, NVars]
        // medians: [Batch, 1, NVars]
        // dec_out: [Batch, PredLen, NVars]
        // Expand stats to [Batch, PredLen, NVars]

        // Burn broadcasting:

        let dec_out = dec_out.mul(stdev); // Broadcast dim 1
        dec_out.add(means)
    }
}

#[cfg(test)]
mod tests {
    use super::{super::test_util::assert_module_forecast, PatchTST, PatchTSTConfig};
    use crate::args::Args;
    use burn::backend::Wgpu;
    use burn::nn::Initializer;
    use clap::Parser;

    #[test]
    fn test_patch_tst_forecast() {
        type B = Wgpu;
        let args = Args::parse_from(vec![
            "test",
            "long-term-forecast",
            "single",
            "ot",
            "time-f",
            "wgpu",
            "--data-path",
            "data/ETT/ETTh1.csv",
            "--skip-training",
        ]);
        let initializer = Initializer::Constant { value: (0.01) };
        let config = PatchTSTConfig::new(args)
            .with_initializer(initializer)
            .init(&device);
        let device = Default::default();
        let model = PatchTST::<B>::new(config, &device);

        assert_module_forecast::<B, PatchTST<B>>(model);
    }
}
