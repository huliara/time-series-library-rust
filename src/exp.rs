pub mod anomaly_detection;
pub mod classification;
pub mod imputation;
pub mod long_term_forecast;
pub mod short_term_forecast;
pub mod zero_shot_forecast;
use crate::args::RootArgs as Args;

pub trait Exp {
    fn train(&mut self, arg: &Args);
    fn validate(&mut self, arg: &Args);
    fn test(&mut self, arg: &Args);
    fn run_exp(&mut self, args: &Args) {
        if !args.skip_training {
            for ii in 0..args.itr {
                println!("Epoch: {}", ii + 1);
                self.train(args);
                self.test(args);
            }
        } else {
            self.test(args);
        }
    }
}

pub fn get_exp_name(args: &Args) -> String {
    format!(
        "{:?}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}",
        args.task_name,
        args.model_id,
        args.model_config,
        args.data_config.data,
        args.data_config.feature_type,
        args.time_lengths.seq_len,
        args.time_lengths.label_len,
        args.time_lengths.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.expand,
        args.d_conv,
        args.factor,
        args.data_config.embed,
        args.distil,
        args.des,
    )
}
