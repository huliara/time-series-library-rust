use clap::ValueEnum;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, ValueEnum, PartialEq, Eq, Deserialize, Serialize)]
pub enum Backend {
    Wgpu,
}
