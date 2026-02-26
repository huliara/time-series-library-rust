use crate::args::{feature_type::FeatureType, target::Target, time_embed::TimeEmbed};
use clap::{Args, ValueEnum};
use core::fmt;
use serde::{Deserialize, Serialize};
#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct DataConfig {
    #[arg(long, value_enum)]
    pub data: Data,
    //corresponds to features
    #[arg(long, value_enum)]
    pub feature_type: FeatureType,

    #[arg(long, value_enum)]
    pub target: Target,

    #[arg(long, value_enum)]
    pub embed: TimeEmbed,

    #[arg(long, default_value = "ETTh1.csv")]
    pub data_path: String,
}
impl Default for DataConfig {
    fn default() -> Self {
        Self {
            data: Data::ETTh1,
            feature_type: FeatureType::Single,
            target: Target::OT,
            embed: TimeEmbed::TimeF,
            data_path: "data/ETT/ETTh1.csv".to_string(),
        }
    }
}

impl fmt::Display for DataConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}_{}", self.data, self.target)
    }
}
#[derive(Debug, Clone, ValueEnum, Deserialize, Serialize)]

pub enum Data {
    ETTh1,
}

impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Data::ETTh1 => "ETTh1",
        };
        write!(f, "{}", s)
    }
}
