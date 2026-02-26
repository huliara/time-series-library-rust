use super::traits::Forecast;
use crate::args::time_lengths::TimeLengths;
use crate::args::{activation::ActivationArg, exp::TaskName};

use crate::layers::{
    embed::patch_embedding::PatchEmbedding,
    embed::patch_embedding::PatchEmbeddingConfig,
    self_attention_family::attention_layer::AttentionLayerConfig,
    self_attention_family::full_attention::FullAttentionConfig,
    transformer_enc_dec::Encoder,
    transformer_enc_dec::{EncoderConfig, EncoderLayerConfig},
};
use burn::{
    config::Config,
    module::Module,
    nn::{BatchNorm, Dropout, DropoutConfig, Initializer, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use serde::{Deserialize, Serialize};

use clap::Args;

#[derive(Debug, Clone, Deserialize, Serialize, Args)]
pub struct PatchTSTArgs {
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
    #[arg(long, value_enum)]
    pub activation: ActivationArg,
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
    pub fn init<B: Backend>(
        &self,
        task_name: TaskName,
        lengths: TimeLengths,
        device: &B::Device,
    ) -> PatchTST<B> {
        let padding = self.model_args.stride;
        let patch_embedding = PatchEmbeddingConfig::new(
            self.model_args.d_model,
            self.model_args.patch_len,
            self.model_args.stride,
            padding,
            5000,
            self.model_args.dropout,
        )
        .with_initializer(self.initializer.clone())
        .init(device);
        let encoder_layer_config = EncoderLayerConfig {
            attention_config: AttentionLayerConfig {
                inner_attention: FullAttentionConfig {
                    mask_flag: false,
                    scale: None,
                    attention_dropout: self.model_args.dropout,
                    output_attention: false,
                },
                d_model: self.model_args.d_model,
                n_heads: self.model_args.n_heads,
                d_keys: None,
                d_values: None,
            },
            d_model: self.model_args.d_model,
            d_ff: Some(self.model_args.d_ff),
            dropout: self.model_args.dropout,
            activation: self.model_args.activation.clone(),
            initializer: self.initializer.clone(),
        };
        let encoder = EncoderConfig::new(
            self.model_args.e_layers,
            encoder_layer_config,
            self.model_args.d_model,
        )
        .with_initializer(self.initializer.clone())
        .init::<B>(device);
        // Prediction Head
        let head_nf = self.model_args.d_model
            * ((lengths.seq_len - self.model_args.patch_len) / self.model_args.stride + 2);

        let head = match task_name {
            TaskName::LongTermForecast | TaskName::ShortTermForecast => Some(
                FlattenHeadConfig::new(head_nf, lengths.pred_len, self.model_args.dropout)
                    .with_initializer(self.initializer.clone())
                    .init(device),
            ),
            TaskName::Imputation | TaskName::AnomalyDetection => Some(
                FlattenHeadConfig::new(head_nf, lengths.seq_len, self.model_args.dropout)
                    .with_initializer(self.initializer.clone())
                    .init(device),
            ),
            _ => None,
        };

        let classification_projection = match task_name {
            TaskName::Classification => Some(
                LinearConfig::new(head_nf * self.model_args.enc_in, self.model_args.num_class)
                    .with_initializer(self.initializer.clone())
                    .init(device),
            ),
            _ => None,
        };

        PatchTST {
            patch_embedding,
            encoder,
            head,
            classification_projection,
        }
    }
}

#[derive(Config, Debug)]
pub struct FlattenHeadConfig {
    pub nf: usize,
    pub target_window: usize,
    pub head_dropout: f64,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}
impl FlattenHeadConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FlattenHead<B> {
        let linear = LinearConfig::new(self.nf, self.target_window)
            .with_initializer(self.initializer.clone())
            .init(device);
        let dropout = DropoutConfig::new(self.head_dropout).init();

        FlattenHead { linear, dropout }
    }
}

#[derive(Module, Debug)]
pub struct FlattenHead<B: Backend> {
    linear: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> FlattenHead<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 3> {
        let x_flat = x.flatten(-2, -1); // [bs, nvars, d_model * patch_num]
        let x_out = self.linear.forward(x_flat);
        self.dropout.forward(x_out)
    }
}

#[derive(Module, Debug)]
pub struct PatchTST<B: Backend> {
    patch_embedding: PatchEmbedding<B>,
    encoder: Encoder<B, BatchNorm<B>>,
    head: Option<FlattenHead<B>>,
    classification_projection: Option<Linear<B>>,
}

impl<B: Backend> Forecast<B> for PatchTST<B> {
    fn forecast(
        &self,
        x_enc: Tensor<B, 3>,
        _x_mark_enc: Tensor<B, 3>,
        _x_dec: Tensor<B, 3>,
        _x_mark_dec: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let means = x_enc.clone().mean_dim(1); // [Batch, 1, NVars]
        let x_enc = x_enc.sub(means.clone()); // Broadcast on dim 1

        let var = x_enc.clone().var(1);
        let stdev = (var + 1e-5).sqrt(); // [Batch, 1, NVars]
        let x_enc = x_enc.div(stdev.clone());
        let x_enc = x_enc.swap_dims(1, 2);

        let (enc_out, n_vars) = self.patch_embedding.forward(x_enc.clone());

        let (enc_out, _) = self.encoder.forward(enc_out, None);
        let enc_out = enc_out.clone().reshape([
            -1isize,
            n_vars.try_into().unwrap(),
            enc_out.dims()[3 - 2].try_into().unwrap(),
            enc_out.dims()[3 - 1].try_into().unwrap(),
        ]);

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
    use crate::args::activation::ActivationArg;
    use crate::args::exp::TaskName;
    use crate::args::time_lengths::TimeLengths;
    use crate::models::patch_tst::PatchTSTArgs;
    use burn::backend::Wgpu;
    use burn::nn::Initializer;

    #[test]
    fn test_patch_tst_forecast() {
        type B = Wgpu;
        let device = Default::default();
        let task_name = TaskName::LongTermForecast;
        let patch_tst_args = PatchTSTArgs {
            num_class: 10,
            d_model: 512,
            patch_len: 16,
            stride: 8,
            enc_in: 7,
            e_layers: 2,
            n_heads: 8,
            d_ff: 2048,
            dropout: 0.1,
            activation: ActivationArg::Gelu,
        };

        let lengths = TimeLengths {
            seq_len: 96,
            pred_len: 96,
            label_len: 48,
        };

        let initializer = Initializer::Constant { value: (0.01) };

        let model = PatchTSTConfig::new(patch_tst_args)
            .with_initializer(initializer)
            .init(task_name, lengths, &device);

        assert_module_forecast::<B, PatchTST<B>>(model);
    }
}
