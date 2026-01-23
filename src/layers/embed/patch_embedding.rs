use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, PositionalEncoding, PositionalEncodingConfig,
    },
    tensor::{backend::Backend, ops::unfold::calculate_unfold_windows, Tensor},
};

use crate::layers::replication_pad_1d::ReplicationPad1d;

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
