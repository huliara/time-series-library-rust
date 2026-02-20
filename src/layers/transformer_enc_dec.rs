use crate::activation::Activation;
use crate::args::activation::ActivationArg;
use crate::layers::self_attention_family::attention_layer::{AttentionLayer, AttentionLayerConfig};
use burn::config::Config;
use burn::module::Module;
use burn::nn::{
    conv::{Conv1d, Conv1dConfig},
    Dropout, DropoutConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::nn::{BatchNorm, BatchNormConfig};
use burn::tensor::{
    activation::{gelu, relu},
    backend::Backend,
    Bool, Tensor,
};

#[derive(Config, Debug)]
pub struct EncoderLayerConfig {
    pub attention_config: AttentionLayerConfig,
    pub d_model: usize,
    pub d_ff: Option<usize>,
    pub dropout: f64,
    pub activation: ActivationArg,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}
impl EncoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> EncoderLayer<B> {
        let d_ff = self.d_ff.unwrap_or(4 * self.d_model);
        let conv1 = Conv1dConfig::new(self.d_model, d_ff, 1)
            .with_initializer(self.initializer.clone())
            .init(device);
        let conv2 = Conv1dConfig::new(d_ff, self.d_model, 1)
            .with_initializer(self.initializer.clone())
            .init(device);
        let norm1 = LayerNormConfig::new(self.d_model).init(device);
        let norm2 = LayerNormConfig::new(self.d_model).init(device);
        let dropout = DropoutConfig::new(self.dropout).init();
        let activation = self.activation.init();
        let attention = self.attention_config.clone().init(device);

        EncoderLayer {
            attention,
            conv1,
            conv2,
            norm1,
            norm2,
            dropout,
            activation,
        }
    }
}

#[derive(Module, Debug)]
pub struct EncoderLayer<B: Backend> {
    attention: AttentionLayer<B>,
    conv1: Conv1d<B>,
    conv2: Conv1d<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    dropout: Dropout,
    activation: Activation,
}

impl<B: Backend> EncoderLayer<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        attn_mask: Option<Tensor<B, 4, Bool>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 4>>) {
        let (new_x, attn) = self
            .attention
            .forward(x.clone(), x.clone(), x.clone(), attn_mask);
        let x = x + self.dropout.forward(new_x);
        let y = self.norm1.forward(x.clone());
        let y = self.conv1.forward(y.transpose());
        let y = self.activation.forward(y);
        let y = self.dropout.forward(y);
        let y = self.conv2.forward(y).transpose();
        let y = self.dropout.forward(y);

        (self.norm2.forward(x + y), attn)
    }
}

#[derive(Config, Debug)]
pub struct EncoderConfig {
    pub n_layers: usize,
    pub layer_config: EncoderLayerConfig,
    pub d_mdel: usize,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}
impl EncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Encoder<B, BatchNorm<B>> {
        let layers: Vec<EncoderLayer<B>> = (0..self.n_layers)
            .map(|_| {
                self.layer_config
                    .clone()
                    .with_initializer(self.initializer.clone())
                    .init(device)
            })
            .collect();

        let norm_layer = Some(BatchNormConfig::new(self.d_mdel).init(device));

        Encoder { layers, norm_layer }
    }
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend, M: Module<B>> {
    layers: Vec<EncoderLayer<B>>,
    norm_layer: Option<M>,
}

impl<B: Backend> Encoder<B, BatchNorm<B>> {
    pub fn forward(
        &self,
        mut x: Tensor<B, 3>,
        attn_mask: Option<Tensor<B, 4, Bool>>,
    ) -> (Tensor<B, 3>, Vec<Option<Tensor<B, 4>>>) {
        let mut attns = Vec::new();
        for layer in &self.layers {
            let (new_x, attn) = layer.forward(x, attn_mask.clone());
            x = new_x;
            attns.push(attn);
        }

        if let Some(norm) = &self.norm_layer {
            x = x.transpose();
            x = norm.forward(x);
            x = x.transpose();
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
        x_mask: Option<Tensor<B, 4, Bool>>,
        cross_mask: Option<Tensor<B, 4, Bool>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 4>>, Option<Tensor<B, 4>>) {
        let (new_x, self_attn) =
            self.self_attention
                .forward(x.clone(), x.clone(), x.clone(), x_mask);
        let x = x + self.dropout.forward(new_x);
        let x = self.norm1.forward(x);

        let (new_x, cross_attn) =
            self.cross_attention
                .forward(x.clone(), cross.clone(), cross.clone(), cross_mask);
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
    pub fn new(
        layers: Vec<DecoderLayer<B>>,
        norm_layer: Option<LayerNorm<B>>,
        projection: Linear<B>,
    ) -> Self {
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
        x_mask: Option<Tensor<B, 4, Bool>>,
        cross_mask: Option<Tensor<B, 4, Bool>>,
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
