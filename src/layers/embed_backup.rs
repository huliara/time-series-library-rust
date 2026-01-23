use crate::args::TimeEmbed;
use crate::layers::replication_pad_1d::ReplicationPad1d;
use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{
    Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig1d, PositionalEncoding,
    PositionalEncodingConfig,
};
use burn::tensor::{backend::Backend, Tensor};
use burn_tensor::ops::unfold::calculate_unfold_windows;

#[derive(Module, Debug)]
pub struct TokenEmbedding<B: Backend> {
    conv: Conv1d<B>,
}

impl<B: Backend> TokenEmbedding<B> {
    pub fn new(c_in: usize, d_model: usize, device: &B::Device) -> Self {
        // Padding 1 for kernel size 3 to keep length same (if stride 1)
        // PyTorch: padding=1, kernel_size=3
        let conv = Conv1dConfig::new(c_in, d_model, 3)
            .with_padding(PaddingConfig1d::Explicit(1))
            .with_bias(false)
            .init(device);
        Self { conv }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // x: [batch, seq_len, c_in]
        // Burn Conv1d expects [batch, channels, length]
        let x = x.swap_dims(1, 2);
        let x = self.conv.forward(x);
        // Return [batch, seq_len, d_model]
        x.swap_dims(1, 2)
    }
}

pub struct TemporalEmbedding<B: Backend> {
    embed: core::marker::PhantomData<B>,
}



#[derive(Module, Debug)]
pub struct DataEmbedding<B: Backend> {
    value_embedding: TokenEmbedding<B>,
    position_embedding: PositionalEncoding<B>,
    // temporal_embedding: TemporalEmbedding<B>, // Skipped for brevity
    dropout: Dropout,
}

impl<B: Backend> DataEmbedding<B> {
    pub fn new(
        c_in: usize,
        d_model: usize,
        _embed_type: TimeEmbed,
        _freq: String,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        let value_embedding = TokenEmbedding::new(c_in, d_model, device);
        let position_embedding = PositionalEncodingConfig::new(d_model)
            .with_max_sequence_size(5000)
            .init(device);

        let temporal_embedding = match _embed_type {
            TimeEmbed::TimeF => Tempo
            TimeEmbed::Fixed => {
                // Implement Fixed Embedding if needed
                // Placeholder for brevity
                unimplemented!()
            }
        };
        let dropout = DropoutConfig::new(dropout).init();

        Self {
            value_embedding,
            position_embedding,
            dropout,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, _x_mark: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let x = self.value_embedding.forward(x);
        let x = self.position_embedding.forward(x);
        self.dropout.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct PatchEmbedding<B: Backend> {
    padding_layer: ReplicationPad1d,
    linear: Linear<B>,
    positional_encoding: PositionalEncoding<B>,
    dropout: Dropout,
    patch_len: usize,
    stride: usize,
}

impl<B: Backend> PatchEmbedding<B> {
    pub fn new(
        d_model: usize,
        patch_len: usize,
        stride: usize,
        padding: usize,
        _dropout: f64,
        device: &B::Device,
    ) -> Self {
        Self {
            padding_layer: ReplicationPad1d::new((0, padding)),
            linear: LinearConfig::new(patch_len, d_model).init(device),
            positional_encoding: PositionalEncodingConfig::new(d_model)
                .with_max_sequence_size(5000)
                .init(device),
            dropout: DropoutConfig::new(_dropout).init(),
            patch_len,
            stride,
        }
    }

    fn unfold(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let dims = x.dims();
        let (batch_size, n_vars, seq_len) = (dims[0], dims[1], dims[2]);

        let num_patches = calculate_unfold_windows(seq_len, self.patch_len, self.stride);

        let mut patches = Vec::with_capacity(num_patches);
        for i in 0..num_patches {
            let start = i * self.stride;
            let end = start + self.patch_len;
            let patch = x.clone().slice([0..batch_size, 0..n_vars, start..end]);
            patches.push(patch);
        }

        let x: Tensor<B, 3> = Tensor::stack(patches, 0);
        x.reshape([batch_size * n_vars, num_patches, self.patch_len])
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, usize) {
        let n_vars = x.dims()[1];
        let x = self.padding_layer.forward(x);
        let x = self.unfold(x);
        let x = self.linear.forward(x);
        let x = self.positional_encoding.forward(x);
        (self.dropout.forward(x), n_vars)
    }
}
