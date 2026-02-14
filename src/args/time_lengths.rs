use clap::Args;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Deserialize, Serialize, Args)]
pub struct TimeLengths {
    #[arg(long, default_value = "96")]
    pub seq_len: usize,
    #[arg(long, default_value = "48")]
    pub label_len: usize,
    #[arg(long, default_value = "96")]
    pub pred_len: usize,
}

impl Default for TimeLengths {
    fn default() -> Self {
        Self {
            seq_len: 96,
            label_len: 48,
            pred_len: 96,
        }
    }
}
