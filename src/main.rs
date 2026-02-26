mod activation;
mod args;
mod data;
mod env_path;
mod exp;
mod layers;
mod models;
mod test_py;

use args::exp::TaskName;
use args::{backend::Backend as ArgBackend, RootArgs};
use burn::backend::{wgpu::WgpuDevice, Autodiff, Wgpu};
use clap::Parser;

use crate::env_path::get_result_root_path;
use crate::exp::long_term_forecast::{infer::infer, train::train};

fn main() {
    let args = RootArgs::parse();

    match args.task_name {
        TaskName::LongTermForecast => {
            if args.backend == ArgBackend::Wgpu {
                type Backend = Autodiff<Wgpu>;
                let device = WgpuDevice::default();
                let result_path = format!(
                    "{}/{}/{}",
                    get_result_root_path(),
                    args.model_config,
                    args.data_config,
                );
                if !args.skip_training {
                    train::<Backend>(
                        &result_path,
                        args.train_config.clone(),
                        args.model_config.clone(),
                        args.data_config.clone(),
                        args.time_lengths.clone(),
                        device.clone(),
                    );
                }
                infer::<Backend>(
                    &result_path,
                    device,
                    args.train_config.clone(),
                    args.model_config.clone(),
                    args.time_lengths.clone(),
                    args.data_config.clone(),
                );
            }
        }
        _ => todo!(),
    };
}
