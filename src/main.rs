mod activation;
mod args;
mod data;
mod exp;
mod layers;
mod models;
mod test_py;

use args::exp::TaskName;
use args::{backend::Backend as ArgBackend, RootArgs};
use burn::backend::{Autodiff, Wgpu};
use clap::Parser;

fn main() {
    let args = RootArgs::parse();
    println!("Args: {:?}", args);

    match args.task_name {
        TaskName::LongTermForecast => {
            if args.backend == ArgBackend::Wgpu {
                type Backend = Autodiff<Wgpu>;
                let device = burn::backend::wgpu::WgpuDevice::default();
                exp::long_term_forecast::train::<Backend>(
                    &args.result_path,
                    args.train_config.clone(),
                    args.model_config.clone(),
                    args.data_config.clone(),
                    args.time_lengths.clone(),
                    device,
                );
            }
        }
        _ => todo!(),
    };
}
