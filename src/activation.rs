use burn::module::Module;
use burn::nn::{Gelu, Relu};
use burn::tensor::{backend::Backend, Tensor};
#[derive(Debug, Module, Clone)]
pub enum Activation {
    ReLu(Relu),
    GeLu(Gelu),
}

impl Activation {
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        match self {
            Activation::ReLu(relu) => relu.forward(input),
            Activation::GeLu(gelu) => gelu.forward(input),
        }
    }
}
