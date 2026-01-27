use burn::{tensor::backend::Backend, Tensor};
pub trait Forward<B: Backend> {
    fn forward(
        &self,
        x: Tensor<B, 3>,
        x_mark: Tensor<B, 3>,
        y: Tensor<B, 3>,
        y_mark: Tensor<B, 3>,
    ) -> Tensor<B, 3>;
}
