use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

use crate::args::time_embed::TimeEmbed;

#[derive(Module, Debug)]
pub struct TimeFeatureEmbedding<B: Backend> {
    embed: Linear<B>,
}

impl<B: Backend> TimeFeatureEmbedding<B> {
    pub fn new(d_model: usize, _embed_type: &TimeEmbed, freq: &str, device: &B::Device) -> Self {
        let d_inp = match freq {
            "h" => 4,
            "t" => 5,
            "s" => 6,
            "m" | "a" => 1,
            "w" => 2,
            "d" | "b" => 3,
            _ => 4, // Default fallback
        };

        let embed = LinearConfig::new(d_inp, d_model)
            .with_bias(false)
            .init(device);

        Self { embed }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.embed.forward(x)
    }
}
