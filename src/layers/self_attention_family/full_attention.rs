use super::triangular_mask::TriangularMask;
use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig},
    prelude::Bool,
    tensor::{activation::softmax, backend::Backend, Tensor},
};
#[derive(Config, Debug)]
pub struct FullAttentionConfig {
    pub mask_flag: bool,
    pub scale: Option<f64>,
    pub attention_dropout: f64,
    pub output_attention: bool,
}

impl FullAttentionConfig {
    pub fn init(&self) -> FullAttention {
        FullAttention {
            scale: self.scale,
            output_attention: self.output_attention,
            dropout: DropoutConfig::new(self.attention_dropout).init(),
            mask_flag: self.mask_flag,
        }
    }
}

#[derive(Module, Debug, Clone)]
pub struct FullAttention {
    scale: Option<f64>,
    output_attention: bool,
    dropout: Dropout,
    mask_flag: bool,
}

impl FullAttention {
    pub fn forward<B: Backend>(
        &self,
        queries: Tensor<B, 4>,
        keys: Tensor<B, 4>,
        values: Tensor<B, 4>,
        attn_mask: Option<Tensor<B, 4, Bool>>,
    ) -> (Tensor<B, 4>, Option<Tensor<B, 4>>) {
        // queries: [B, L, H, E]
        // keys: [B, S, H, E]
        // values: [B, S, H, D]

        let [b, l, _, e] = queries.dims();

        let scale = self.scale.unwrap_or(1.0 / (e as f64).sqrt());

        // queries: [B, L, H, E] -> [B, H, L, E]
        let queries_perm = queries.clone().permute([0, 2, 1, 3]);
        // keys: [B, S, H, E] -> [B, H, E, S]
        let keys_perm = keys.clone().permute([0, 2, 3, 1]);

        let mut scores = queries_perm.matmul(keys_perm);

        if self.mask_flag {
            let mask: Tensor<B, 4, Bool> = attn_mask.unwrap_or(TriangularMask::<B>::new(b, l).mask);
            scores = scores.mask_fill(mask, -f64::INFINITY); // [B, H, L, S]
        }
        let attn = softmax(scores * scale, 3);
        let attn = self.dropout.forward(attn);

        // values: [B, S, H, D] -> [B, H, S, D]
        let values_perm = values.permute([0, 2, 1, 3]);
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

#[cfg(test)]
mod tests {
    use super::*;

    use burn::tensor::Shape;
    use burn_ndarray::NdArray;

    #[test]
    fn test_full_attention_forward() {
        type B = NdArray;
        let device = Default::default();

        let attention = FullAttentionConfig::new(true, 0.1, true).init();

        let b_size = 2;
        let l = 4;
        let s = 4;
        let h = 8;
        let e = 32;
        let d = 32;

        let queries = Tensor::<B, 4>::zeros([b_size, l, h, e], &device);
        let keys = Tensor::<B, 4>::zeros([b_size, s, h, e], &device);
        let values = Tensor::<B, 4>::zeros([b_size, s, h, d], &device);

        let (out, attn) = attention.forward(queries, keys, values, None);

        assert_eq!(out.shape(), Shape::new([b_size, l, h, d]));

        if let Some(attn_tensor) = attn {
            assert_eq!(attn_tensor.shape(), Shape::new([b_size, h, l, s]));
        } else {
            panic!("Expected attention output");
        }
    }
}
