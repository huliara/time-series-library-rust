mod activation;
mod args;
mod data;
mod exp;
mod layers;
mod models;
mod test_py;
use args::exp::TaskName;
use args::RootArgs;
use clap::Parser;

fn main() {
    let args: RootArgs = RootArgs::parse();
    println!("Args: {:?}", args);

    match args.task_name {
        TaskName::AnomalyDetection => todo!(),
        TaskName::Classification => todo!(), // run_exp(ExpClassification { args }),
        TaskName::Imputation => todo!(),     // run_exp(ExpImputation { args }),
        TaskName::LongTermForecast => run_exp(ExpLongTermForecast { args }),
        TaskName::ShortTermForecast => todo!(), // run_exp(ExpShortTermForecast { args }),
        TaskName::ZeroShotForecast => todo!(),  // run_exp(ExpZeroShotForecast { args }),
    };
}
