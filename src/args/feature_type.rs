use clap::ValueEnum;
use core::fmt;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq, Deserialize, Serialize)]
pub enum FeatureType {
    Single,
    Multi,
}
impl fmt::Display for FeatureType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            FeatureType::Single => "single",
            FeatureType::Multi => "multi",
        };
        write!(f, "{}", s)
    }
}
