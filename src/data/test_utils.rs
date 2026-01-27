use clap::Parser;
use std::sync::Arc;

use crate::args::Args;
use crate::data::{
    batcher::TimeSeriesBatch, data_loader::create_data_loader, dataset::ett_hour::ExpFlag,
};
use burn::{data::dataloader::DataLoader, tensor::backend::Backend};

pub fn setup_test_dataloader<B: Backend>() -> Arc<dyn DataLoader<B, TimeSeriesBatch<B>>> {
    let args = Args::parse_from(vec![
        "test",
        "long-term-forecast",
        "single",
        "ot",
        "time-f",
        "wgpu",
        "--data-path",
        "data/ETT/ETTh1.csv",
        "--skip-training",
    ]);

    create_data_loader::<B>(&args, ExpFlag::Test)
}
