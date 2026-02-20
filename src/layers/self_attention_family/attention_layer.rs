use crate::layers::self_attention_family::full_attention::FullAttentionConfig;

use super::full_attention::FullAttention;
use burn::config::Config;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::Bool;
use burn::tensor::{backend::Backend, Tensor};

#[derive(Config, Debug)]
pub struct AttentionLayerConfig {
    pub inner_attention: FullAttentionConfig,
    pub d_model: usize,
    pub n_heads: usize,
    pub d_keys: Option<usize>,
    pub d_values: Option<usize>,
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

#[cfg(test)]
mod tests {
    use super::*;

    use burn::tensor::Shape;
    use burn_ndarray::NdArray;

    #[test]
    fn test_attention_layer_forward() {
        type B = NdArray;
        let device = Default::default();

        let inner_config = FullAttentionConfig {
            mask_flag: false,
            scale: None,
            attention_dropout: 0.1,
            output_attention: true,
        };

        let config = AttentionLayerConfig {
            inner_attention: inner_config,
            d_model: 32,
            n_heads: 4,
            d_keys: None,
            d_values: None,
        };
        let attention_layer = config.init::<B>(&device);

        let b_size = 2;
        let l = 8;
        let s = 8;
        let d_model = 32;

        let queries = Tensor::<B, 3>::zeros([b_size, l, d_model], &device);
        let keys = Tensor::<B, 3>::zeros([b_size, s, d_model], &device);
        let values = Tensor::<B, 3>::zeros([b_size, s, d_model], &device);

        let (out, attn) = attention_layer.forward(queries, keys, values, None);

        assert_eq!(out.shape(), Shape::new([b_size, l, d_model]));

        if let Some(attn_tensor) = attn {
            assert_eq!(
                attn_tensor.shape(),
                Shape::new([b_size, config.n_heads, l, s])
            );
        } else {
            panic!("Expected attention output");
        }
    }
}
