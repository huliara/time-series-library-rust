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
#[derive(Debug, Clone, Deserialize, Serialize, Args, Default)]
pub struct PatchTSTArgs {
    #[arg(long, default_value_t = 96)]
    pub seq_len: usize,
    #[arg(long, default_value_t = 96)]
    pub pred_len: usize,
    #[arg(long, default_value_t = 10)]
    pub num_class: usize,
    #[arg(long, default_value_t = 512)]
    pub d_model: usize,
    #[arg(long, default_value_t = 16)]
    pub patch_len: usize,
    #[arg(long, default_value_t = 8)]
    pub stride: usize,
    #[arg(long, default_value_t = 7)]
    pub enc_in: usize,
    #[arg(long, default_value_t = 2)]
    pub e_layers: usize,
    #[arg(long, default_value_t = 8)]
    pub n_heads: usize,
    #[arg(long, default_value_t = 2048)]
    pub d_ff: usize,
    #[arg(long, default_value_t = 0.1)]
    pub dropout: f64,
    #[arg(long, default_value_t = 1)]
    pub factor: usize,
    #[arg(long, default_value = "gelu")]
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
        let padding = self.model_args.stride;
        let patch_embedding = PatchEmbedding::new(
            self.model_args.d_model,
            self.model_args.patch_len,
            self.model_args.stride,
            padding,
            self.model_args.dropout,
            device,
        );

        // Encoder
        // Create Encoder Layers
        let mut layers = Vec::new();
        for _ in 0..self.model_args.e_layers {
            let attn_layer = AttentionLayer::new(
                FullAttention::new(
                    false,
                    self.model_args.factor,
                    None,
                    self.model_args.dropout,
                    false,
                ),
                self.model_args.d_model,
                self.model_args.n_heads,
                None,
                None,
                device,
            );

            let layer = EncoderLayer::new(
                attn_layer,
                self.model_args.d_model,
                Some(self.model_args.d_ff),
                self.model_args.dropout,
                self.model_args.activation.clone(),
                device,
            );
            layers.push(layer);
        }

        let encoder = Encoder::new(layers, None);

        // Prediction Head
        let head_nf = &self.model_args.d_model
            * ((&self.model_args.seq_len - &self.model_args.patch_len) / &self.model_args.stride
                + 2);

        let head = if task_name == TaskName::LongTermForecast
            || task_name == TaskName::ShortTermForecast
        {
            Some(FlattenHead::new(
                head_nf,
                self.model_args.pred_len,
                self.model_args.dropout,
                device,
            ))
        } else if task_name == TaskName::Imputation || task_name == TaskName::AnomalyDetection {
            Some(FlattenHead::new(
                head_nf,
                self.model_args.seq_len,
                self.model_args.dropout,
                device,
            ))
        } else {
            None
        };

        let classification_projection = if task_name == TaskName::Classification {
            Some(
                LinearConfig::new(head_nf * &self.model_args.enc_in, self.model_args.num_class)
                    .init(device),
            )
        } else {
            None
        };

        PatchTST {
            patch_embedding,
            encoder,
            head,
            classification_projection,
            seq_len: self.model_args.seq_len,
            pred_len: self.model_args.pred_len,
            num_class: self.model_args.num_class,
            d_model: self.model_args.d_model,
            patch_len: self.model_args.patch_len,
            stride: self.model_args.stride,
            enc_in: self.model_args.enc_in,
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

    // &self.model_args
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
    use crate::models::patch_tst::PatchTSTArgs;
    use burn::backend::Wgpu;
    use burn::nn::Initializer;

    #[test]
    fn test_patch_tst_forecast() {
        type B = Wgpu;
        let device = Default::default();
        let task_name = crate::args::TaskName::LongTermForecast;
        let args = PatchTSTArgs::default();
        let initializer = Initializer::Constant { value: (0.01) };
        let model = PatchTSTConfig::new(args)
            .with_initializer(initializer)
            .init(task_name, &device);

        assert_module_forecast::<B, PatchTST<B>>(model);
    }
}
