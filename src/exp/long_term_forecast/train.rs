use crate::{
    args::{data_config::DataConfig, model_config::ModelConfig, time_lengths::TimeLengths},
    data::{
        batcher::TimeSeriesBatcher,
        dataset::ett_hour::{ETTHourDataset, ExpFlag},
    },
    exp::{create_artifact_dir, long_term_forecast::ForecastModel},
};
use burn::{
    data::dataloader::DataLoaderBuilder,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{Learner, SupervisedTraining},
};
use clap::Args;
use serde::{Deserialize, Serialize};
#[derive(Debug, Args, Clone, Deserialize, Serialize)]
pub struct ExpConfig {
    #[arg(long, default_value_t = 10)]
    pub num_epochs: usize,
    #[arg(long, default_value_t = 64)]
    pub batch_size: usize,
    #[arg(long, default_value_t = 4)]
    pub num_workers: usize,
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    #[arg(long, default_value_t = 1.0e-4)]
    pub learning_rate: f64,
}

pub fn train<B>(
    artifact_dir: &str,
    train_config: ExpConfig,
    model_config: ModelConfig,
    data_config: DataConfig,
    lengths: TimeLengths,
    device: B::Device,
) where
    B: AutodiffBackend,
{
    create_artifact_dir(artifact_dir);

    B::seed(&device, train_config.seed);

    let batcher = TimeSeriesBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(train_config.batch_size)
        .shuffle(train_config.seed)
        .num_workers(train_config.num_workers)
        .build(ETTHourDataset::new(
            &data_config,
            &lengths,
            ExpFlag::Train,
            &device,
        ));

    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(train_config.batch_size)
        .shuffle(train_config.seed)
        .num_workers(train_config.num_workers)
        .build(ETTHourDataset::new(
            &data_config,
            &lengths,
            ExpFlag::_Val,
            &device,
        ));

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_valid)
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(train_config.num_epochs)
        .summary();
    let optimizer = AdamConfig::new().init();
    let model = ForecastModel::<B>::new(model_config, &device);
    let result = training.launch(Learner::new(model, optimizer, train_config.learning_rate));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
