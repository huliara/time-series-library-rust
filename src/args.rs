use core::fmt;

use clap::{Parser, ValueEnum};

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
pub enum TaskName {
    AnomalyDetection,
    Classification,
    Imputation,
    LongTermForecast,
    ShortTermForecast,
    ZeroShotForecast,
}

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
pub enum Backend {
    Wgpu,
}
#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
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

#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
pub enum Target {
    HUFL,
    HULL,
    MUFL,
    MULL,
    LUFL,
    LULL,
    OT,
}
impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Target::HUFL => "HUFL",
            Target::HULL => "HULL",
            Target::MUFL => "MUFL",
            Target::MULL => "MULL",
            Target::LUFL => "LUFL",
            Target::LULL => "LULL",
            Target::OT => "OT",
        };
        write!(f, "{}", s)
    }
}
#[derive(Debug, Clone, ValueEnum, PartialEq, Eq)]
pub enum TimeEmbed {
    TimeF,
    Fixed,
}
impl fmt::Display for TimeEmbed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            TimeEmbed::TimeF => "timeF",
            TimeEmbed::Fixed => "fixed",
        };
        write!(f, "{}", s)
    }
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(value_enum)]
    pub task_name: TaskName,

    #[arg(value_enum)]
    pub feature_type: FeatureType,

    #[arg(value_enum)]
    pub target: Target,

    #[arg(value_enum)]
    pub embed: TimeEmbed,

    #[arg(value_enum)]
    pub backend: Backend,

    #[arg(long)]
    pub skip_training: bool,

    #[arg(long, default_value = "test")]
    pub model_id: String,

    #[arg(long, default_value = "Autoformer")]
    pub model: String,

    // data loader
    #[arg(long, default_value = "ETTh1")]
    pub data: String,

    #[arg(long, default_value = "./data/ETT/")]
    pub root_path: String,

    #[arg(long, default_value = "ETTh1.csv")]
    pub data_path: String,

    #[arg(long, default_value = "h")]
    pub freq: String,

    #[arg(long, default_value = "./result/")]
    pub result_path: String,
    // forecasting task
    #[arg(long, default_value_t = 96)]
    pub seq_len: usize,

    #[arg(long, default_value_t = 48)]
    pub label_len: usize,

    #[arg(long, default_value_t = 96)]
    pub pred_len: usize,

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

    #[arg(long, default_value = "gelu")]
    pub activation: String,

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
    // GPU
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

    #[arg(long, default_value = "./checkpoints/")]
    pub checkpoints: String,
}
