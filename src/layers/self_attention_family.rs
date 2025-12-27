use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::{activation::softmax, backend::Backend, Tensor};
use std::marker::PhantomData;

#[derive(Module, Debug)]
pub struct FullAttention<B: Backend> {
    scale: Option<f64>,
    mask_flag: bool,
    output_attention: bool,
    dropout: Dropout,
    factor: usize,
    _phantom: PhantomData<B>,
}

impl<B: Backend> FullAttention<B> {
    pub fn new(
        mask_flag: bool,
        factor: usize,
        scale: Option<f64>,
        attention_dropout: f64,
        output_attention: bool,
    ) -> Self {
        Self {
            scale,
            mask_flag,
            output_attention,
            dropout: DropoutConfig::new(attention_dropout).init(),
            factor,
            _phantom: PhantomData,
        }
    }

    pub fn forward(
        &self,
        queries: Tensor<B, 4>,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
        _attn_mask: Option<Tensor<B, 4>>,
    ) -> (Tensor<B, 4>, Option<Tensor<B, 4>>) {
        // queries: [B, L, H, E]
        // keys: [B, S, H, E]
        // values: [B, S, H, D]

        let [_, _, _, e] = queries.dims();

        let scale = self.scale.unwrap_or(1.0 / (e as f64).sqrt());

        // queries: [B, L, H, E] -> [B, H, L, E]
        let queries_perm = queries.clone().permute([0, 2, 1, 3]);
        // keys: [B, S, H, E] -> [B, H, E, S]
        let keys_perm = keys.clone().permute([0, 2, 3, 1]);

        let scores = queries_perm.matmul(keys_perm); // [B, H, L, S]
        let scores = scores * scale;

        // TODO: Implement masking
        // if let Some(mask) = attn_mask { ... }

        let attn = softmax(scores, 3);
        let attn = self.dropout.forward(attn);

        // values: [B, S, H, D] -> [B, H, S, D]
        let values_perm = values.clone().permute([0, 2, 1, 3]);
        let out = attn.clone().matmul(values_perm); // [B, H, L, D]

        // [B, H, L, D] -> [B, L, H, D]
        let out = out.permute([0, 2, 1, 3]);

        if self.output_attention {
            (out, Some(attn))
        } else {
            (out, None)
        }
    }
}

#[derive(Module, Debug)]
pub struct AttentionLayer<B: Backend> {
    inner_attention: FullAttention<B>,
    query_projection: Linear<B>,
    key_projection: Linear<B>,
    value_projection: Linear<B>,
    out_projection: Linear<B>,
    n_heads: usize,
}

impl<B: Backend> AttentionLayer<B> {
    pub fn new(
        inner_attention: FullAttention<B>,
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
        attn_mask: Option<Tensor<B, 4>>,
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
