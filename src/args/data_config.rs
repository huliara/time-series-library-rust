use crate::args::{feature_type::FeatureType, target::Target, time_embed::TimeEmbed};
use clap::Args;
use serde::{Deserialize, Serialize};
#[derive(Args, Debug, Clone, Deserialize, Serialize)]
pub struct DataConfig {
    #[arg(long, default_value = "ETTh1")]
    pub data: String,
    //corresponds to features
    #[arg(long, value_enum)]
    pub feature_type: FeatureType,

    #[arg(long, value_enum)]
    pub target: Target,

    #[arg(long, value_enum)]
    pub embed: TimeEmbed,

    #[arg(long, default_value = "./data/ETT/")]
    pub root_path: String,

    #[arg(long, default_value = "ETTh1.csv")]
    pub data_path: String,
}
impl Default for DataConfig {
    fn default() -> Self {
        Self {
            data: "ETTh1".to_string(),
            feature_type: FeatureType::Single,
            target: Target::OT,
            embed: TimeEmbed::TimeF,
            root_path: "./".to_string(),
            data_path: "data/ETT/ETTh1.csv".to_string(),
        }
    }
}
