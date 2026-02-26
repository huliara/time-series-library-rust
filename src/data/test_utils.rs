use std::sync::Arc;

use crate::{
    args::{data_config::DataConfig, time_lengths::TimeLengths},
    data::{batcher::TimeSeriesBatch, data_loader::create_data_loader, dataset::ett_hour::ExpFlag},
};
use burn::{data::dataloader::DataLoader, tensor::backend::Backend};

pub fn setup_test_dataloader<B: Backend>() -> Arc<dyn DataLoader<B, TimeSeriesBatch<B>>> {
    let data_config = DataConfig::default();
    let lengths = TimeLengths::default();
    let batch_size = 32;
    let seed = 42;
    let num_workers = 0;
    create_data_loader::<B>(
        &data_config,
        &lengths,
        batch_size,
        num_workers,
        seed,
        ExpFlag::Test,
    )
}
