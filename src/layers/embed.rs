use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Dropout, DropoutConfig, PaddingConfig1d};
use burn::tensor::{backend::Backend, Tensor};
use burn_tensor::s;

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

#[derive(Module, Debug)]
pub struct PositionalEmbedding<B: Backend> {
    d_model: usize,
    max_len: usize,
    pe: Tensor<B, 3>, // [1, max_len, d_model]
}

impl<B: Backend> PositionalEmbedding<B> {
    pub fn new(d_model: usize, max_len: usize, device: &B::Device) -> Self {
        let mut pe: Tensor<B, 2> = Tensor::zeros([max_len, d_model], device);
        // Implement actual sinusoidal position encoding initialization
        let position: Tensor<B, 2> = Tensor::arange_step(0..max_len as i64, 2, device)
            .float()
            .unsqueeze_dim(1); // [max_len, 1]

        let div_term: Tensor<B, 2> = (Tensor::arange_step(0..d_model as i64, 2, device).float()
            * -((10000.0f64).ln() / d_model as f64).exp())
        .unsqueeze_dim(0); // [1, d_model/2]

        let theta = position * div_term; // [max_len, d_model/2]
        pe = pe.slice_assign([0..max_len, (0..d_model).s], theta.sin());
        pe = pe.slice_assign(s![0..max_len,1..;2], theta.cos());
        pe = pe.unsqueeze_dim(0); // [1, max_len, d_model]
        let result: Tensor<B, 3> = pe.unsqueeze_dim(0);
        Self {
            d_model,
            max_len,
            pe: result,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let seq_len = x.dims()[1];
        if seq_len > self.max_len {
            panic!("Sequence length exceeds maximum length");
        }
        self.pe.clone().slice([0..1, 0..seq_len, 0..self.d_model])
    }
}

#[derive(Module, Debug)]
pub struct DataEmbedding<B: Backend> {
    value_embedding: TokenEmbedding<B>,
    position_embedding: PositionalEmbedding<B>,
    // temporal_embedding: TemporalEmbedding<B>, // Skipped for brevity
    dropout: Dropout,
}

impl<B: Backend> DataEmbedding<B> {
    pub fn new(
        c_in: usize,
        d_model: usize,
        _embed_type: String,
        _freq: String,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        let value_embedding = TokenEmbedding::new(c_in, d_model, device);
        let position_embedding = PositionalEmbedding::new(d_model, 5000, device);
        let dropout = DropoutConfig::new(dropout).init();

        Self {
            value_embedding,
            position_embedding,
            dropout,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, _x_mark: Option<Tensor<B, 3>>) -> Tensor<B, 3> {
        let x = self.value_embedding.forward(x);
        let pe = self.position_embedding.forward(x.clone());

        let x = x + pe;
        self.dropout.forward(x)
    }
}
