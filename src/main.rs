mod args;
mod data;
mod exp;
mod layers;
mod models;
use args::Args;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{Autodiff, Wgpu};
use clap::Parser;
use exp::anomaly_detection::ExpAnomalyDetection;
// use exp::classification::ExpClassification;
// use exp::imputation::ExpImputation;
use exp::long_term_forecast::ExpLongTermForecast;
// use exp::short_term_forecast::ExpShortTermForecast;
// use exp::zero_shot_forecast::ExpZeroShotForecast;
use models::transformer::Transformer;

fn main() {
    let args: Args = Args::parse();
    println!("Args: {:?}", args);

    type Backend = Wgpu;
    type AutodiffBackend = Autodiff<Backend>;
    let device = WgpuDevice::default();

    if args.model == "Transformer" {
        let _model: Transformer<Backend> = Transformer::new(&args, &device);
        println!("Model initialized successfully.");
        // println!("Model: {:?}", model); // Debug print might be too large
    } else {
        println!(
            "Model {} not implemented yet. Please use --model Transformer",
            args.model
        );
    }

    match args.task_name {
        args::TaskName::AnomalyDetection => todo!(),
        args::TaskName::Classification => todo!(), // run_exp(ExpClassification { args }),
        args::TaskName::Imputation => todo!(),     // run_exp(ExpImputation { args }),
        args::TaskName::LongTermForecast => {
            ExpLongTermForecast::new(args.clone())
                .train::<AutodiffBackend, Transformer<AutodiffBackend>>(&args);
        }
        args::TaskName::ShortTermForecast => todo!(), // run_exp(ExpShortTermForecast { args }),
        args::TaskName::ZeroShotForecast => todo!(),  // run_exp(ExpZeroShotForecast { args }),
    };
}
