use burn::{
    module::Module,
    tensor::{backend::Backend, s, Int, Tensor},
};

#[derive(Module, Debug)]
pub struct PositionalEmbedding<B: Backend> {
    pe: Tensor<B, 3>,
}

impl<B: Backend> PositionalEmbedding<B> {
    pub fn new(d_model: usize, max_len: usize, device: &B::Device) -> Self {
        let position: Tensor<B, 2> = Tensor::<B, 1, Int>::arange(0..max_len as i64, device)
            .float()
            .unsqueeze_dim(1); // [max_len, 1]

        let div_term = Tensor::<B, 1, Int>::arange_step(0..(d_model as i64), 2, device)
            .float()
            .mul_scalar(-(10000.0f32.ln()) / d_model as f32)
            .exp()
            .unsqueeze_dim(0);

        let term = position.matmul(div_term); // [max_len, ceil(d_model/2)]

        let pe_sin = term.clone().sin().unsqueeze_dim(2); // [max_len, d_model/2, 1]
        let pe_cos = term.clone().cos().unsqueeze_dim(2); // [max_len, d_model/2, 1]

        let mut pre_pe = Tensor::<B, 2>::zeros(term.shape(), device);

        pre_pe = pre_pe.slice_assign(s![..,0..d_model;2], pe_sin);
        pre_pe = pre_pe.slice_assign(s![..,1..d_model;2], pe_cos);

        let pe = pre_pe.unsqueeze_dim(0); // [1, max_len, d_model]

        Self { pe }
    }

    pub fn forward(&self, x: &Tensor<B, 3>) -> Tensor<B, 3> {
        let seq_len = x.dims()[1];
        let [_b, _l, d] = x.dims();
        // Slice length and d_model to ensure matching shapes if needed, mostly for length
        self.pe.clone().slice([0..1, 0..seq_len, 0..d])
    }
}
