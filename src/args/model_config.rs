use crate::{
    args::{exp::TaskName, time_lengths::TimeLengths},
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
