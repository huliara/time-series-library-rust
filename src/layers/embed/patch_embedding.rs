use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Initializer, Linear, LinearConfig, PositionalEncoding,
        PositionalEncodingConfig,
    },
    tensor::{backend::Backend, Tensor},
};

use crate::layers::replication_pad_1d::ReplicationPad1d;

#[derive(Config, Debug)]
pub struct PatchEmbeddingConfig {
    pub d_model: usize,
    pub patch_len: usize,
    pub stride: usize,
    pub padding: usize,
    pub dropout: f64,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl PatchEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PatchEmbedding<B> {
        PatchEmbedding {
            padding_layer: ReplicationPad1d::new((0, self.padding)),
            linear: LinearConfig::new(self.patch_len, self.d_model)
                .with_initializer(self.initializer.clone())
                .init(device),
            positional_encoding: PositionalEncodingConfig::new(self.d_model)
                .with_max_sequence_size(5000)
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            patch_len: self.patch_len,
            stride: self.stride,
        }
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
    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, usize) {
        let n_vars = x.dims()[1];
        let x = self.padding_layer.forward(x);
        let x: Tensor<B, 4> = x.unfold(-1, self.patch_len, self.stride);
        let x = x
            .clone()
            .reshape([x.dims()[0] * x.dims()[1], x.dims()[2], x.dims()[3]]);
        let x = self.linear.forward(x);
        let x = self.positional_encoding.forward(x);
        (self.dropout.forward(x), n_vars)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Tensor};

    #[test]
    fn test_patch_embedding_forward() {
        let config = PatchEmbeddingConfig::new(16, 4, 2, 2, 0.1);

        let device = burn::backend::wgpu::WgpuDevice::default();
        let patch_embedding = config.init(&device);

        let x = Tensor::<burn::backend::wgpu::Wgpu, 3>::random(
            [2, 3, 10],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let (output, n_vars) = patch_embedding.forward(x);

        assert_eq!(output.dims(), [2 * 3, 5, 16]); // batch_size * n_vars, num_patches, d_model
        assert_eq!(n_vars, 3);
    }
}
