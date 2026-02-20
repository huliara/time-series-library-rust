use burn::{
    config::Config,
    module::Module,
    tensor::{backend::Backend, s, Int, Shape, Tensor},
};

#[derive(Config, Debug)]
pub struct PositionalEmbeddingConfig {
    pub d_model: usize,
    pub max_len: usize,
}

impl PositionalEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PositionalEmbedding<B> {
        let position: Tensor<B, 2> = Tensor::<B, 1, Int>::arange(0..self.max_len as i64, device)
            .float()
            .unsqueeze_dim(1); // [max_len, 1]
        let div_term = Tensor::arange_step(0..(self.d_model as i64), 2, device)
            .float()
            .mul_scalar(-(10000.0f32.ln()) / self.d_model as f32)
            .exp()
            .unsqueeze_dim(0);
        let term = position.mul(div_term); // [max_len, ceil(d_model/2)]
        let pe_sin = term.clone().sin(); // [max_len, d_model/2, 1]
        let pe_cos = term.clone().cos(); // [max_len, d_model/2, 1]
        let mut pre_pe = Tensor::<B, 2>::zeros(Shape::new([self.max_len, self.d_model]), device);

        pre_pe = pre_pe.slice_assign(s![..,0..self.d_model;2], pe_sin);
        pre_pe = pre_pe.slice_assign(s![..,1..self.d_model;2], pe_cos);

        let pe = pre_pe.unsqueeze_dim(0); // [1, max_len, d_model]

        PositionalEmbedding { pe }
    }
}

#[derive(Module, Debug)]
pub struct PositionalEmbedding<B: Backend> {
    pe: Tensor<B, 3>,
}

impl<B: Backend> PositionalEmbedding<B> {
    pub fn forward(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, seq_len, _d_model] = x.dims();
        self.pe.clone().slice(s![.., 0..seq_len])
    }
}

#[cfg(test)]
mod tests {
    use burn_ndarray::NdArray;

    use super::*;

    type TestBackend = NdArray;

    #[test]
    fn test_positional_embedding_forward() {
        let device = Default::default();
        let d_model = 32;
        let max_len = 100;
        let pe = PositionalEmbeddingConfig::new(d_model, max_len).init(&device);

        let batch_size = 2;
        let seq_len = 50;
        let x = Tensor::<TestBackend, 3>::zeros([batch_size, seq_len, d_model], &device);

        // xのshapeに基づいてpeがスライスされて返される
        let output = pe.forward(&x);

        // 出力のshapeを確認: [1, seq_len, d_model]
        // batchサイズはブロードキャスト用に1になる
        assert_eq!(output.dims(), [1, seq_len, d_model]);
    }
}
