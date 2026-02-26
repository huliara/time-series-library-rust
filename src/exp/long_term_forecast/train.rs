use crate::{
    args::{data_config::DataConfig, model_config::ModelConfig, time_lengths::TimeLengths},
    data::{data_loader::create_data_loader, dataset::ett_hour::ExpFlag},
    exp::{create_artifact_dir, long_term_forecast::ForecastModel, Train},
};
use burn::{
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

impl<B: AutodiffBackend> Train<B> for ForecastModel<B> {
    fn train(
        &self,
        result_path: &str,
        exp_config: ExpConfig,
        model_config: ModelConfig,
        data_config: DataConfig,
        lengths: TimeLengths,
        device: B::Device,
    ) where
        B: AutodiffBackend,
    {
        create_artifact_dir(result_path);

        B::seed(&device, exp_config.seed);

        let dataloader_train = create_data_loader(
            &data_config,
            &lengths,
            exp_config.batch_size,
            exp_config.num_workers,
            exp_config.seed,
            ExpFlag::Train,
        );

        let dataloader_valid = create_data_loader(
            &data_config,
            &lengths,
            exp_config.batch_size,
            exp_config.num_workers,
            exp_config.seed,
            ExpFlag::Val,
        );

        let training = SupervisedTraining::new(result_path, dataloader_train, dataloader_valid)
            .with_file_checkpointer(CompactRecorder::new())
            .num_epochs(exp_config.num_epochs)
            .summary();
        let optimizer = AdamConfig::new().init();
        let model = ForecastModel::<B>::new(model_config, lengths, &device);
        let result = training.launch(Learner::new(model, optimizer, exp_config.learning_rate));

        result
            .model
            .save_file(format!("{result_path}/model"), &CompactRecorder::new())
            .expect("Trained model should be saved successfully");
    }
}
