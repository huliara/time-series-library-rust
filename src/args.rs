pub mod activation;
pub mod backend;
pub mod data_config;
pub mod exp;
pub mod feature_type;
pub mod model_config;
pub mod target;
pub mod time_embed;
pub mod time_lengths;
use self::exp::TaskName;
use self::time_lengths::TimeLengths;
use crate::{
    args::{backend::Backend, data_config::DataConfig, model_config::ModelConfig},
    exp::long_term_forecast::train::ExpConfig,
};
use clap::Parser;
use serde::{Deserialize, Serialize};
#[derive(Parser, Debug, Clone, Deserialize, Serialize)]
#[command(name = "exp")]
#[command(author, version, about, long_about = None)]
pub struct RootArgs {
    #[command(subcommand)]
    pub model_config: ModelConfig,
    #[arg(long, value_enum)]
    pub task_name: TaskName,
    #[arg(long, value_enum)]
    pub backend: Backend,
    #[command(flatten)]
    pub data_config: DataConfig,
    #[command(flatten)]
    pub time_lengths: TimeLengths,
    #[command(flatten)]
    pub exp_config: ExpConfig,

    #[arg(long)]
    pub skip_training: bool,

    #[arg(long, default_value = "test")]
    pub model_id: String,

    #[arg(long, default_value = "./checkpoints/")]
    pub checkpoints: String,
}
