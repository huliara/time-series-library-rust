use burn::{
    module::{Module, Param},
    nn::{Embedding, EmbeddingConfig},
    tensor::{backend::Backend, s, Int, Tensor},
};

#[derive(Module, Debug)]
pub struct FixedEmbedding<B: Backend> {
    pub emb: Embedding<B>,
}

impl<B: Backend> FixedEmbedding<B> {
    pub fn new(c_in: usize, d_model: usize, device: &B::Device) -> Self {
        let position: Tensor<B, 2> = Tensor::<B, 1, Int>::arange(0..c_in as i64, device)
            .float()
            .unsqueeze_dim(1);

        let div_term = Tensor::<B, 1, Int>::arange_step(0..(d_model as i64), 2, device)
            .float()
            .mul_scalar(-(10000.0f32.ln()) / d_model as f32)
            .exp()
            .unsqueeze_dim(0);

        let term = position.matmul(div_term);

        let pe_sin = term.clone().sin().unsqueeze_dim(2);
        let pe_cos = term.clone().cos().unsqueeze_dim(2);

        let mut pre_pe = Tensor::<B, 2>::zeros(term.shape(), device);

        pre_pe = pre_pe.slice_assign(s![..,0..d_model;2], pe_sin);
        pre_pe = pre_pe.slice_assign(s![..,1..d_model;2], pe_cos);
        let pe = pre_pe.unsqueeze_dim(0);

        let mut emb = EmbeddingConfig::new(c_in, d_model).init(device);
        emb.weight = Param::from_tensor(pe);

        Self { emb }
    }

    pub fn forward(&self, x: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.emb.forward(x).detach()
    }
}
