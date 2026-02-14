use crate::models::patch_tst::PatchTSTArgs;
use clap::Subcommand;
use core::fmt;
use serde::{Deserialize, Serialize};
#[derive(Subcommand, Debug, Clone, Deserialize, Serialize)]
pub enum ModelConfig {
    PatchTST(PatchTSTArgs),
    Transformer,
    // Other model configs can be added here
}
impl fmt::Display for ModelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ModelConfig::PatchTST(_) => "PatchTST",
            ModelConfig::Transformer => "Transformer",
        };
        write!(f, "{}", s)
    }
}
