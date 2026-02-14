use crate::{
    args::time_embed::TimeEmbed,
    layers::embed::{
        positional_embedding::{PositionalEmbedding, PositionalEmbeddingConfig},
        temporal_embedding::TemporalEmbedding,
        time_feature_embedding::TimeFeatureEmbedding,
        token_embedding::TokenEmbedding,
    },
};
use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug)]
pub enum TemporalEmbed<B: Backend> {
    Temporal(TemporalEmbedding<B>),
    TimeFeature(TimeFeatureEmbedding<B>),
}

impl<B: Backend> TemporalEmbed<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        match self {
            Self::Temporal(m) => m.forward(x),
            Self::TimeFeature(m) => m.forward(x),
        }
    }
}

#[derive(Module, Debug)]
pub struct DataEmbedding<B: Backend> {
    value_embedding: TokenEmbedding<B>,
    position_embedding: PositionalEmbedding<B>,
    temporal_embedding: TemporalEmbed<B>,
    dropout: Dropout,
}

impl<B: Backend> DataEmbedding<B> {
    pub fn new(
        c_in: usize,
        d_model: usize,
        embed_type: TimeEmbed,
        freq: String,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        let value_embedding = TokenEmbedding::new(c_in, d_model, device);
        let position_embedding = PositionalEmbeddingConfig::new(d_model, 5000).init(device);

        let temporal_embedding = if embed_type != TimeEmbed::TimeF {
            TemporalEmbed::Temporal(TemporalEmbedding::new(d_model, &embed_type, &freq, device))
        } else {
            TemporalEmbed::TimeFeature(TimeFeatureEmbedding::new(
                d_model,
                &embed_type,
                &freq,
                device,
            ))
        };

        let dropout = DropoutConfig::new(dropout).init();

        Self {
            value_embedding,
            position_embedding,
            temporal_embedding,
            dropout,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, x_mark: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let val = self.value_embedding.forward(x);
        let pos = self.position_embedding.forward(&val);

        // val and pos Should be [Batch, Seq, d_model]

        let mut out = val + pos;

        if let Some(mark) = x_mark {
            // mark: [Batch, Seq, Features]
            out = out + self.temporal_embedding.forward(mark);
        }

        self.dropout.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct DataEmbeddingInverted<B: Backend> {
    value_embedding: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> DataEmbeddingInverted<B> {
    pub fn new(
        c_in: usize,
        d_model: usize,
        _embed_type: String,
        _freq: String,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        // c_in here is expected to be seq_len (input size of Linear)
        let value_embedding = LinearConfig::new(c_in, d_model).init(device);
        let dropout = DropoutConfig::new(dropout).init();
        Self {
            value_embedding,
            dropout,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, x_mark: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        // x: [Batch, Seq, Variate]
        let x = x.permute([0, 2, 1]); // [Batch, Variate, Seq]

        let inp = if let Some(mark) = x_mark {
            // mark: [Batch, Seq, TimeFeatures]
            let mark = mark.permute([0, 2, 1]); // [Batch, TimeFeatures, Seq]
            Tensor::cat(vec![x, mark], 1) // [Batch, Variate+TimeFeatures, Seq]
        } else {
            x
        };

        let out = self.value_embedding.forward(inp); // [Batch, Var, d_model]
        self.dropout.forward(out)
    }
}

#[derive(Module, Debug)]
pub struct DataEmbeddingWoPos<B: Backend> {
    value_embedding: TokenEmbedding<B>,
    // position_embedding: PositionalEmbedding<B>, // Created but unused in Python? We skip it here to avoid unused field.
    temporal_embedding: TemporalEmbed<B>,
    dropout: Dropout,
}

impl<B: Backend> DataEmbeddingWoPos<B> {
    pub fn new(
        c_in: usize,
        d_model: usize,
        embed_type: TimeEmbed,
        freq: String,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        let value_embedding = TokenEmbedding::new(c_in, d_model, device);
        // let position_embedding = PositionalEmbedding::new(d_model, 5000, device);

        let temporal_embedding = if embed_type != TimeEmbed::TimeF {
            TemporalEmbed::Temporal(TemporalEmbedding::new(d_model, &embed_type, &freq, device))
        } else {
            TemporalEmbed::TimeFeature(TimeFeatureEmbedding::new(
                d_model,
                &embed_type,
                &freq,
                device,
            ))
        };

        let dropout = DropoutConfig::new(dropout).init();

        Self {
            value_embedding,
            temporal_embedding,
            dropout,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, x_mark: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let val = self.value_embedding.forward(x);
        let mut out = val;

        if let Some(mark) = x_mark {
            out = out + self.temporal_embedding.forward(mark);
        }

        self.dropout.forward(out)
    }
}
