use std::sync::Arc;

use crate::args::Args;

use crate::data::batcher::TimeSeriesBatch;
use crate::data::{
    batcher::TimeSeriesBatcher,
    dataset::ett_hour::{ETTHourDataset, ExpFlag},
};
use burn::data::dataloader::DataLoader;
use burn::{data::dataloader::DataLoaderBuilder, prelude::Backend};

pub fn create_data_loader<B: Backend>(
    args: &Args,
    flag: ExpFlag,
) -> Arc<dyn DataLoader<B, TimeSeriesBatch<B>>> {
    let device = B::Device::default();
    let dataset: ETTHourDataset<B> = ETTHourDataset::new(args, flag, &device);
    DataLoaderBuilder::new(TimeSeriesBatcher::default())
        .batch_size(args.batch_size)
        .shuffle(args.seed)
        .build(dataset)
}
