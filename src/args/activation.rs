use std::fmt;

use crate::activation::Activation;
use burn::nn::{Gelu, Relu};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq, Deserialize, Serialize, Default)]
pub enum ActivationArg {
    #[default]
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

impl fmt::Display for ActivationArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ActivationArg::Relu => "relu",
            ActivationArg::Gelu => "gelu",
        };
        write!(f, "{}", s)
    }
}
