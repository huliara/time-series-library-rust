pub mod long_term_forecast;
use crate::args::RootArgs as Args;

pub fn get_exp_name(args: &Args) -> String {
    format!(
        "{:?}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}",
        args.task_name,
        args.model_config,
        args.data_config.data,
        args.data_config.feature_type,
        args.time_lengths.seq_len,
        args.time_lengths.label_len,
        args.time_lengths.pred_len,
        args.data_config.embed,
    )
}
fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}
