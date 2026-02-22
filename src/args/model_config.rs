use crate::{
    args::exp::TaskName,
    models::{
        dlinear::{DLinearArgs, DLinearConfig},
        patch_tst::{PatchTSTArgs, PatchTSTConfig},
        traits::Forecast,
    },
};
use clap::Subcommand;
use core::fmt;
use serde::{Deserialize, Serialize};
#[derive(Subcommand, Debug, Clone, Deserialize, Serialize)]
pub enum ModelConfig {
    PatchTST(PatchTSTArgs),
    DLinear(DLinearArgs),
    // Other model configs can be added here
}
impl fmt::Display for ModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ModelConfig::PatchTST(_) => "PatchTST",
            ModelConfig::DLinear(_) => "DLinear",
        };
        write!(f, "{}", s)
    }
}

impl ModelConfig {
    pub fn init<B: burn::tensor::backend::Backend>(
        &self,
        task_name: TaskName,
        device: &B::Device,
    ) -> Box<dyn Forecast<B>> {
        match self {
            ModelConfig::PatchTST(args) => {
                Box::new(PatchTSTConfig::new(args.clone()).init(task_name, device))
            }
            ModelConfig::DLinear(args) => {
                Box::new(DLinearConfig::new(args.clone()).init(task_name, device))
            }
        }
    }
}
