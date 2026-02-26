use crate::activation::Activation;
use burn::nn::{Gelu, Relu};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, ValueEnum, PartialEq, Eq, Deserialize, Serialize, Default, strum::Display,
)]
pub enum ActivationArg {
    #[default]
    #[strum(serialize = "relu")]
    Relu,
    #[strum(serialize = "gelu")]
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
