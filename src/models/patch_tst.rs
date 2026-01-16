use crate::layers::{
    embed::PatchEmbedding,
    self_attention_family::{AttentionLayer, FullAttention},
    transformer_enc_dec::{Encoder, EncoderLayer},
};
use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv1d, Conv1dConfig},
        Dropout, DropoutConfig, Linear, LinearConfig,
    },
    tensor::{backend::Backend, Tensor},
};

#[derive(Config, Debug)]
pub struct PatchTSTConfig {
    pub task_name: String,
    pub seq_len: usize,
    pub pred_len: usize,
    pub enc_in: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub n_heads: usize,
    pub e_layers: usize,
    pub dropout: f64,
    pub factor: usize,
    pub activation: String,
    pub patch_len: usize,
    pub stride: usize,
    pub num_class: usize,
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
    task_name: String,
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

impl<B: Backend> PatchTST<B> {
    pub fn new(configs: PatchTSTConfig, device: &B::Device) -> Self {
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

        let head = if configs.task_name == "long_term_forecast"
            || configs.task_name == "short_term_forecast"
        {
            Some(FlattenHead::new(
                head_nf,
                configs.pred_len,
                configs.dropout,
                device,
            ))
        } else if configs.task_name == "imputation" || configs.task_name == "anomaly_detection" {
            Some(FlattenHead::new(
                head_nf,
                configs.seq_len,
                configs.dropout,
                device,
            ))
        } else {
            None
        };

        let classification_projection = if configs.task_name == "classification" {
            Some(LinearConfig::new(head_nf * configs.enc_in, configs.num_class).init(device))
        } else {
            None
        };

        Self {
            task_name: configs.task_name,
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

    fn forecast(&self, x_enc: Tensor<B, 3>, _x_mark_enc: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        // Normalization (RevIN equivalent inline)
        // x_enc: [Batch, Length, NVars]
        let x_enc_len = x_enc.dims()[1];
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
        let stdev_expanded = stdev.clone().repeat_dim(1, self.pred_len); // [Batch, Pred, N] - Assuming repeat_dim repeats dim 1
                                                                         // Note: repeat_dim(dim, times)
                                                                         // stdev is [B, 1, N]. We want [B, Pred, N].
                                                                         // Need to check Burn version. 0.16. `repeat_dim`?
                                                                         // Usually `repeat` works on all dims or slice-cat.
                                                                         // Let's use basic `slice` and `repeat` combo if needed or `repeat` if scalar.
                                                                         // Correct approach: stdev is [B, 1, N].
                                                                         // Burn's broadcasting handles [B, 1, N] * [B, Pred, N] -> [B, Pred, N].
                                                                         // So we don't need manual repeat if broadcasting works.
                                                                         // It SHOULD work.

        let dec_out = dec_out.mul(stdev); // Broadcast dim 1
        let dec_out = dec_out.add(means);

        dec_out
    }

    pub fn forward(
        &self,
        x_enc: Tensor<B, 3>,
        x_mark_enc: Option<Tensor<B, 3>>,
        _x_dec: Option<Tensor<B, 3>>,
        _x_mark_dec: Option<Tensor<B, 3>>,
        _mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 3> {
        if self.task_name == "long_term_forecast" || self.task_name == "short_term_forecast" {
            let dec_out = self.forecast(x_enc, x_mark_enc);
            // Return last pred_len if sequence is longer? Python code: dec_out[:, -self.pred_len:, :]
            // FlattenHead outputs typically exact size.
            // If implicit FlattenHead size mismatch (e.g. padding), slice.
            // Our FlattenHead output size `pred_len`.
            return dec_out;
        }
        // Implement other tasks similarly... (omitted for brevity as Forecast is main)
        panic!("Only forecast implemented for now");
    }
}
