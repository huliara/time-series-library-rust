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
use crate::args::{backend::Backend, data_config::DataConfig, model_config::ModelConfig};
use clap::Parser;
use serde::{Deserialize, Serialize};
#[derive(Parser, Debug, Clone, Deserialize, Serialize)]
#[command(name = "exp")]
#[command(author, version, about, long_about = None)]
pub struct RootArgs {
    #[arg(long, value_enum)]
    pub task_name: TaskName,
    #[command(flatten)]
    pub data_config: DataConfig,
    #[command(flatten)]
    pub time_lengths: TimeLengths,

    #[arg(long, value_enum)]
    pub backend: Backend,

    #[arg(long)]
    pub skip_training: bool,

    #[command(subcommand)]
    pub model_config: ModelConfig,

    #[arg(long, default_value = "test")]
    pub model_id: String,

    #[arg(long, default_value = "h")]
    pub freq: String,

    #[arg(long, default_value = "./result/")]
    pub result_path: String,

    #[arg(long, default_value = "Monthly")]
    pub seasonal_patterns: String,

    #[arg(long, default_value_t = false)]
    pub inverse: bool,

    // imputation task
    #[arg(long, default_value_t = 0.25)]
    pub mask_rate: f32,

    // anomaly detection task
    #[arg(long, default_value_t = 0.25)]
    pub anomaly_ratio: f32,

    // model define
    #[arg(long, default_value_t = 2)]
    pub expand: usize,

    #[arg(long, default_value_t = 4)]
    pub d_conv: usize,

    #[arg(long, default_value_t = 5)]
    pub top_k: usize,

    #[arg(long, default_value_t = 6)]
    pub num_kernels: usize,

    #[arg(long, default_value_t = 7)]
    pub enc_in: usize,

    #[arg(long, default_value_t = 7)]
    pub dec_in: usize,

    #[arg(long, default_value_t = 7)]
    pub c_out: usize,

    #[arg(long, default_value_t = 512)]
    pub d_model: usize,

    #[arg(long, default_value_t = 8)]
    pub n_heads: usize,

    #[arg(long, default_value_t = 2)]
    pub e_layers: usize,

    #[arg(long, default_value_t = 1)]
    pub d_layers: usize,

    #[arg(long, default_value_t = 2048)]
    pub d_ff: usize,

    #[arg(long, default_value_t = 25)]
    pub moving_avg: usize,

    #[arg(long, default_value_t = 1)]
    pub factor: usize,

    #[arg(long, default_value_t = true)]
    pub distil: bool,

    #[arg(long, default_value_t = 0.1)]
    pub dropout: f64,

    #[arg(long, default_value_t = 1)]
    pub channel_independence: i32,

    #[arg(long, default_value = "moving_avg")]
    pub decomp_method: String,

    #[arg(long, default_value_t = 1)]
    pub use_norm: i32,

    #[arg(long, default_value_t = 0)]
    pub down_sampling_layers: usize,

    #[arg(long, default_value_t = 1)]
    pub down_sampling_window: usize,

    #[arg(long)]
    pub down_sampling_method: Option<String>,

    #[arg(long, default_value_t = 96)]
    pub seg_len: usize,

    // optimization
    #[arg(long, default_value_t = 10)]
    pub num_workers: usize,

    #[arg(long, default_value_t = 1)]
    pub itr: usize,

    #[arg(long, default_value_t = 10)]
    pub train_epochs: usize,

    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,

    #[arg(long, default_value_t = 3)]
    pub patience: usize,

    #[arg(long, default_value_t = 0.0001)]
    pub learning_rate: f64,

    #[arg(long, default_value = "test")]
    pub des: String,

    #[arg(long, default_value = "MSE")]
    pub loss: String,

    #[arg(long, default_value = "type1")]
    pub lradj: String,

    #[arg(long, default_value_t = false)]
    pub use_amp: bool,

    #[arg(long, default_value_t = true)]
    pub use_gpu: bool,

    #[arg(long, default_value_t = 0)]
    pub gpu: i32,

    #[arg(long, default_value = "cuda")]
    pub gpu_type: String,

    #[arg(long, default_value_t = false)]
    pub use_multi_gpu: bool,

    #[arg(long, default_value = "0,1,2,3")]
    pub devices: String,

    // de-stationary projector params
    #[arg(long, num_args = 1.., default_values_t = vec![128, 128])]
    pub p_hidden_dims: Vec<i32>,

    #[arg(long, default_value_t = 2)]
    pub p_hidden_layers: i32,

    // metrics (dtw)
    #[arg(long, default_value_t = false)]
    pub use_dtw: bool,

    // Augmentation
    #[arg(long, default_value_t = 0)]
    pub augmentation_ratio: i32,

    #[arg(long, default_value_t = 2)]
    pub seed: u64,

    #[arg(long, default_value_t = false)]
    pub jitter: bool,

    #[arg(long, default_value_t = false)]
    pub scaling: bool,

    #[arg(long, default_value_t = false)]
    pub permutation: bool,

    #[arg(long, default_value_t = false)]
    pub randompermutation: bool,

    #[arg(long, default_value_t = false)]
    pub magwarp: bool,

    #[arg(long, default_value_t = false)]
    pub timewarp: bool,

    #[arg(long, default_value_t = false)]
    pub windowslice: bool,

    #[arg(long, default_value_t = false)]
    pub windowwarp: bool,

    #[arg(long, default_value_t = false)]
    pub rotation: bool,

    #[arg(long, default_value_t = false)]
    pub spawner: bool,

    #[arg(long, default_value_t = false)]
    pub dtwwarp: bool,

    #[arg(long, default_value_t = false)]
    pub shapedtwwarp: bool,

    #[arg(long, default_value_t = false)]
    pub wdba: bool,

    #[arg(long, default_value_t = false)]
    pub discdtw: bool,

    #[arg(long, default_value_t = false)]
    pub discsdtw: bool,

    #[arg(long, default_value = "")]
    pub extra_tag: String,

    // TimeXer
    #[arg(long, default_value_t = 16)]
    pub patch_len: usize,

    // GCN
    #[arg(long, default_value_t = 10)]
    pub node_dim: usize,

    #[arg(long, default_value_t = 2)]
    pub gcn_depth: usize,

    #[arg(long, default_value_t = 0.3)]
    pub gcn_dropout: f64,

    #[arg(long, default_value_t = 0.3)]
    pub propalpha: f64,

    #[arg(long, default_value_t = 32)]
    pub conv_channel: usize,

    #[arg(long, default_value_t = 32)]
    pub skip_channel: usize,

    #[arg(long, default_value_t = false)]
    pub individual: bool,

    // TimeFilter
    #[arg(long, default_value_t = 0.1)]
    pub alpha: f64,

    #[arg(long, default_value_t = 0.5)]
    pub top_p: f64,

    #[arg(long, default_value_t = 1)]
    pub pos: i32,
    #[arg(long, default_value_t = 10)]
    pub num_class: usize,

    #[arg(long, default_value = "./checkpoints/")]
    pub checkpoints: String,
}
