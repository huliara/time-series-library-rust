use crate::layers::self_attention_family::full_attention::FullAttentionConfig;

use super::full_attention::FullAttention;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::Bool;
use burn::tensor::{backend::Backend, Tensor};

#[derive(Config, Debug)]
pub struct AttentionLayerConfig {
    pub inner_attention: FullAttentionConfig,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_keys: Option<usize>,
    pub d_values: Option<usize>,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl AttentionLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> AttentionLayer<B> {
        AttentionLayer::new(
            self.inner_attention.init(),
            self.d_model,
            self.n_heads,
            self.d_keys,
            self.d_values,
            device,
        )
    }
}

#[derive(Module, Debug)]
pub struct AttentionLayer<B: Backend> {
    inner_attention: FullAttention,
    query_projection: Linear<B>,
    key_projection: Linear<B>,
    value_projection: Linear<B>,
    out_projection: Linear<B>,
    n_heads: usize,
}

impl<B: Backend> AttentionLayer<B> {
    pub fn new(
        inner_attention: FullAttention,
        d_model: usize,
        n_heads: usize,
        d_keys: Option<usize>,
        d_values: Option<usize>,
        device: &B::Device,
    ) -> Self {
        let d_keys = d_keys.unwrap_or(d_model / n_heads);
        let d_values = d_values.unwrap_or(d_model / n_heads);

        let query_projection = LinearConfig::new(d_model, d_keys * n_heads).init(device);
        let key_projection = LinearConfig::new(d_model, d_keys * n_heads).init(device);
        let value_projection = LinearConfig::new(d_model, d_values * n_heads).init(device);
        let out_projection = LinearConfig::new(d_values * n_heads, d_model).init(device);

        Self {
            inner_attention,
            query_projection,
            key_projection,
            value_projection,
            out_projection,
            n_heads,
        }
    }

    pub fn forward(
        &self,
        queries: Tensor<B, 3>,
        keys: Tensor<B, 3>,
        values: Tensor<B, 3>,
        attn_mask: Option<Tensor<B, 4, Bool>>,
    ) -> (Tensor<B, 3>, Option<Tensor<B, 4>>) {
        let [b, l, _] = queries.dims();
        let [_, s, _] = keys.dims();
        let h = self.n_heads;

        let queries = self.query_projection.forward(queries);
        let [_, _, d_q_proj] = queries.dims();
        let queries = queries.reshape([b, l, h, d_q_proj / h]);

        let keys = self.key_projection.forward(keys);
        let [_, _, d_k_proj] = keys.dims();
        let keys = keys.reshape([b, s, h, d_k_proj / h]);

        let values = self.value_projection.forward(values);
        let [_, _, d_v_proj] = values.dims();
        let values = values.reshape([b, s, h, d_v_proj / h]);

        let (out, attn) = self
            .inner_attention
            .forward(queries, keys, values, attn_mask);

        let [_, _, _, d_v] = out.dims();
        let out = out.reshape([b, l, h * d_v]);
        let out = self.out_projection.forward(out);

        (out, attn)
    }
}
