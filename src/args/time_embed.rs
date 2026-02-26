use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, ValueEnum, PartialEq, Eq, Deserialize, Serialize, Default, strum::Display,
)]
pub enum TimeEmbed {
    #[default]
    #[strum(serialize = "timeF")]
    TimeF,
    #[strum(serialize = "fixed")]
    Fixed,
}
