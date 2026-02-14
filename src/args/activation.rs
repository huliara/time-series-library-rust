use crate::activation::Activation;
use burn::nn::{Gelu, Relu};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq, Deserialize, Serialize)]
pub enum ActivationArg {
    Relu,
    Gelu,
}

impl ActivationArg {
    pub fn init(&self) -> Activation {
        match self {
            ActivationArg::Relu => Activation::ReLu(Relu),
            ActivationArg::Gelu => Activation::GeLu(Gelu),
        }
    }
}
