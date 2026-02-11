mod activation;
mod args;
mod data;
mod exp;
mod layers;
mod models;
mod test_py;
use args::Args;
use burn::backend::{Autodiff, Wgpu};
use clap::Parser;
// use exp::classification::ExpClassification;
// use exp::imputation::ExpImputation;
use exp::long_term_forecast::ExpLongTermForecast;
// use exp::short_term_forecast::ExpShortTermForecast;
// use exp::zero_shot_forecast::ExpZeroShotForecast;
use models::patch_tst::PatchTST;

fn main() {
    let args: Args = Args::parse();
    println!("Args: {:?}", args);

    type Backend = Wgpu;
    type AutodiffBackend = Autodiff<Backend>;

    match args.task_name {
        args::TaskName::AnomalyDetection => todo!(),
        args::TaskName::Classification => todo!(), // run_exp(ExpClassification { args }),
        args::TaskName::Imputation => todo!(),     // run_exp(ExpImputation { args }),
        args::TaskName::LongTermForecast => {
            ExpLongTermForecast::new(args.clone())
                .train::<AutodiffBackend, PatchTST<AutodiffBackend>>(&args);
        }
        args::TaskName::ShortTermForecast => todo!(), // run_exp(ExpShortTermForecast { args }),
        args::TaskName::ZeroShotForecast => todo!(),  // run_exp(ExpZeroShotForecast { args }),
    };
}
