use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, ValueEnum, PartialEq, Eq, Deserialize, Serialize, Default, strum::Display,
)]
pub enum Target {
    HUFL,
    HULL,
    MUFL,
    MULL,
    LUFL,
    LULL,
    #[default]
    OT,
}
