mod activation;
mod args;
mod data;
mod exp;
mod layers;
mod models;
mod test_py;
use args::exp::TaskName;
use args::RootArgs;
use burn::backend::{Autodiff, Wgpu};
use clap::Parser;
use exp::long_term_forecast::ExpLongTermForecast;
use models::patch_tst::PatchTST;

fn main() {
    let args: RootArgs = RootArgs::parse();
    println!("Args: {:?}", args);

    type Backend = Wgpu;
    type AutodiffBackend = Autodiff<Backend>;

    match args.task_name {
        TaskName::AnomalyDetection => todo!(),
        TaskName::Classification => todo!(), // run_exp(ExpClassification { args }),
        TaskName::Imputation => todo!(),     // run_exp(ExpImputation { args }),
        TaskName::LongTermForecast => {
            ExpLongTermForecast::new(args.clone())
                .train::<AutodiffBackend, PatchTST<AutodiffBackend>>(&args);
        }
        TaskName::ShortTermForecast => todo!(), // run_exp(ExpShortTermForecast { args }),
        TaskName::ZeroShotForecast => todo!(),  // run_exp(ExpZeroShotForecast { args }),
    };
}
