use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(ValueEnum, Debug, Clone, Deserialize, Serialize)]
pub enum TaskName {
    AnomalyDetection,
    Classification,
    Imputation,
    LongTermForecast,
    ShortTermForecast,
    ZeroShotForecast,
}
