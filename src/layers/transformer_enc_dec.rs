use burn::module::Module;
use burn::tensor::{backend::Backend, Tensor, activation::{relu, gelu}};
use burn::nn::{Linear, LinearConfig, LayerNorm, LayerNormConfig, Dropout, DropoutConfig};
use crate::layers::self_attention_family::AttentionLayer;

#[derive(Module, Debug)]
pub struct EncoderLayer<B: Backend> {
    attention: AttentionLayer<B>,
    conv1: Linear<B>,
    conv2: Linear<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
    activation: String,
}

impl<B: Backend> EncoderLayer<B> {
    pub fn new(
        attention: AttentionLayer<B>,
        d_model: usize,
        d_ff: Option<usize>,
        dropout: f64,
        activation: String,
        device: &B::Device,
    ) -> Self {
        let d_ff = d_ff.unwrap_or(4 * d_model);
        let conv1 = LinearConfig::new(d_model, d_ff).init(device);
        let conv2 = LinearConfig::new(d_ff, d_model).init(device);
        let norm1 = LayerNormConfig::new(d_model).init(device);
        let norm2 = LayerNormConfig::new(d_model).init(device);
        let dropout = DropoutConfig::new(dropout).init();

        Self {
            attention,
            conv1,
            conv2,
            norm1,
            norm2,
            dropout,
            activation,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, attn_mask: Option<Tensor<B, 4>>) -> (Tensor<B, 3>, Option<Tensor<B, 4>>) {
        let (new_x, attn) = self.attention.forward(x.clone(), x.clone(), x.clone(), attn_mask);
        let x = x + self.dropout.forward(new_x);

        let y = self.norm1.forward(x.clone());
        let y = self.conv1.forward(y);
        let y = if self.activation == "relu" {
            relu(y)
        } else {
            gelu(y)
        };
        let y = self.dropout.forward(y);
        let y = self.conv2.forward(y);
        let y = self.dropout.forward(y);

        (self.norm2.forward(x + y), attn)
    }
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    layers: Vec<EncoderLayer<B>>,
    norm: Option<LayerNorm<B>>,
}

impl<B: Backend> Encoder<B> {
    pub fn new(layers: Vec<EncoderLayer<B>>, norm_layer: Option<LayerNorm<B>>) -> Self {
        Self {
            layers,
            norm: norm_layer,
        }
    }

    pub fn forward(&self, mut x: Tensor<B, 3>, attn_mask: Option<Tensor<B, 4>>) -> (Tensor<B, 3>, Vec<Option<Tensor<B, 4>>>) {
        let mut attns = Vec::new();
        for layer in &self.layers {
            let (new_x, attn) = layer.forward(x, attn_mask.clone());
            x = new_x;
            attns.push(attn);
        }

        if let Some(norm) = &self.norm {
            x = norm.forward(x);
        }

        (x, attns)
    }
}

#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    self_attention: AttentionLayer<B>,
    cross_attention: AttentionLayer<B>,
    conv1: Linear<B>,
    conv2: Linear<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    norm3: LayerNorm<B>,
    dropout: Dropout,
    activation: String,
}

impl<B: Backend> DecoderLayer<B> {
    pub fn new(
        self_attention: AttentionLayer<B>,
        cross_attention: AttentionLayer<B>,
        d_model: usize,
        d_ff: Option<usize>,
        dropout: f64,
        activation: String,
        device: &B::Device,
    ) -> Self {
        let d_ff = d_ff.unwrap_or(4 * d_model);
        let conv1 = LinearConfig::new(d_model, d_ff).init(device);
        let conv2 = LinearConfig::new(d_ff, d_model).init(device);
        let norm1 = LayerNormConfig::new(d_model).init(device);
        let norm2 = LayerNormConfig::new(d_model).init(device);
        let norm3 = LayerNormConfig::new(d_model).init(device);
        let dropout = DropoutConfig::new(dropout).init();

        Self {
            self_attention,
            cross_attention,
            conv1,
            conv2,
            norm1,
            norm2,
            norm3,
            dropout,
            activation,
        }
    }

    pub fn forward(
        &self, 
        x: Tensor<B, 3>, 
        cross: Tensor<B, 3>, 
        x_mask: Option<Tensor<B, 4>>, 
        cross_mask: Option<Tensor<B, 4>>
    ) -> (Tensor<B, 3>, Option<Tensor<B, 4>>, Option<Tensor<B, 4>>) {
        let (new_x, self_attn) = self.self_attention.forward(x.clone(), x.clone(), x.clone(), x_mask);
        let x = x + self.dropout.forward(new_x);
        let x = self.norm1.forward(x);

        let (new_x, cross_attn) = self.cross_attention.forward(x.clone(), cross.clone(), cross.clone(), cross_mask);
        let x = x + self.dropout.forward(new_x);
        let x = self.norm2.forward(x);

        let y = x.clone();
        let y = self.conv1.forward(y);
        let y = if self.activation == "relu" {
            relu(y)
        } else {
            gelu(y)
        };
        let y = self.dropout.forward(y);
        let y = self.conv2.forward(y);
        let y = self.dropout.forward(y);

        (self.norm3.forward(x + y), self_attn, cross_attn)
    }
}

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    layers: Vec<DecoderLayer<B>>,
    norm: Option<LayerNorm<B>>,
    projection: Linear<B>,
}

impl<B: Backend> Decoder<B> {
    pub fn new(layers: Vec<DecoderLayer<B>>, norm_layer: Option<LayerNorm<B>>, projection: Linear<B>) -> Self {
        Self {
            layers,
            norm: norm_layer,
            projection,
        }
    }

    pub fn forward(
        &self, 
        mut x: Tensor<B, 3>, 
        cross: Tensor<B, 3>, 
        x_mask: Option<Tensor<B, 4>>, 
        cross_mask: Option<Tensor<B, 4>>
    ) -> Tensor<B, 3> {
        for layer in &self.layers {
            let (new_x, _, _) = layer.forward(x, cross.clone(), x_mask.clone(), cross_mask.clone());
            x = new_x;
        }

        if let Some(norm) = &self.norm {
            x = norm.forward(x);
        }

        self.projection.forward(x)
    }
}
