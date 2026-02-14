use crate::{args::time_embed::TimeEmbed, layers::embed::fixed_embedding::FixedEmbedding};
use burn::{
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Int, Tensor},
};

#[derive(Module, Debug)]
pub enum EmbeddingLayer<B: Backend> {
    Fixed(FixedEmbedding<B>),
    Learnable(Embedding<B>),
}

impl<B: Backend> EmbeddingLayer<B> {
    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        match self {
            Self::Fixed(e) => e.forward(x),
            Self::Learnable(e) => e.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct TemporalEmbedding<B: Backend> {
    minute_embed: Option<EmbeddingLayer<B>>,
    hour_embed: EmbeddingLayer<B>,
    weekday_embed: EmbeddingLayer<B>,
    day_embed: EmbeddingLayer<B>,
    month_embed: EmbeddingLayer<B>,
}

impl<B: Backend> TemporalEmbedding<B> {
    pub fn new(d_model: usize, embed_type: &TimeEmbed, freq: &str, device: &B::Device) -> Self {
        let minute_size = 4;
        let hour_size = 24;
        let weekday_size = 7;
        let day_size = 32;
        let month_size = 13;

        let create_embed = |size: usize| -> EmbeddingLayer<B> {
            match embed_type {
                TimeEmbed::Fixed => {
                    EmbeddingLayer::Fixed(FixedEmbedding::new(size, d_model, device))
                }

                TimeEmbed::TimeF => {
                    EmbeddingLayer::Learnable(EmbeddingConfig::new(size, d_model).init(device))
                }
            }
        };

        let minute_embed = if freq == "t" {
            Some(create_embed(minute_size))
        } else {
            None
        };

        Self {
            minute_embed,
            hour_embed: create_embed(hour_size),
            weekday_embed: create_embed(weekday_size),
            day_embed: create_embed(day_size),
            month_embed: create_embed(month_size),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x_int = x.int();

        // Use reshape to remove trailing dimension instead of squeeze if squeeze args are tricky in this version
        let [b, s, _] = x_int.dims();
        let get_slice = |idx: usize| {
            x_int
                .clone()
                .slice([0..b, 0..s, idx..idx + 1])
                .reshape([b, s])
        };

        let hour_x = self.hour_embed.forward(get_slice(3));
        let weekday_x = self.weekday_embed.forward(get_slice(2));
        let day_x = self.day_embed.forward(get_slice(1));
        let month_x = self.month_embed.forward(get_slice(0));

        let mut x_out = hour_x + weekday_x + day_x + month_x;

        if let Some(minute) = &self.minute_embed {
            x_out = x_out + minute.forward(get_slice(4));
        }

        x_out
    }
}
