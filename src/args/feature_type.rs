use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(
    Debug, Clone, ValueEnum, PartialEq, Eq, Deserialize, Serialize, Default, strum::Display,
)]
pub enum FeatureType {
    #[default]
    #[strum(serialize = "single")]
    Single,
    #[strum(serialize = "multi")]
    Multi,
}
