mod args;
mod data;
mod exp;
mod layers;
mod models;

use args::Args;
use burn::backend::wgpu::WgpuDevice;
use burn::backend::Wgpu;
use clap::Parser;
use models::transformer::Transformer;

fn main() {
    let args = Args::parse();
    println!("Args: {:?}", args);

    type Backend = Wgpu;
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
}
