use burn::{
    prelude::Bool,
    tensor::{backend::Backend, Tensor},
};
pub struct TriangularMask<B: Backend> {
    pub mask: Tensor<B, 4, Bool>,
}
impl<B: Backend> TriangularMask<B> {
    pub fn new(batch_size: usize, size: usize) -> Self {
        let mask =
            Tensor::<B, 4, Bool>::tril_mask([batch_size, 1, size, size], 0, &B::Device::default());
        Self { mask }
    }
}
