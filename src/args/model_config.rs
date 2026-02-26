use crate::{
    args::exp::TaskName,
    models::{
        dlinear::{DLinearArgs, DLinearConfig},
        patch_tst::{PatchTSTArgs, PatchTSTConfig},
        traits::Forecast,
    },
};
use clap::Subcommand;
use serde::{Deserialize, Serialize};
#[derive(Subcommand, Debug, Clone, Deserialize, Serialize, strum::Display)]
pub enum ModelConfig {
    #[strum(serialize = "PatchTST")]
    PatchTST(PatchTSTArgs),
    #[strum(serialize = "DLinear")]
    DLinear(DLinearArgs),
    // Other model configs can be added here
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
