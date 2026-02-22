mod forecast_output;
use crate::{
    args::{
        data_config::{self, DataConfig},
        exp::TaskName,
        model_config::{self, ModelConfig},
        time_lengths::TimeLengths,
        RootArgs,
    },
    data::{
        batcher::{TimeSeriesBatch, TimeSeriesBatcher},
        data_loader::create_data_loader,
        dataset::ett_hour::{ETTHourDataset, ExpFlag},
    },
    exp::long_term_forecast::forecast_output::ForecastOutput,
    models::{
        dlinear::{DLinear, DLinearConfig},
        patch_tst::{PatchTST, PatchTSTConfig},
        traits::Forecast,
    },
};
use burn::{
    data::dataloader::DataLoaderBuilder,
    module::AutodiffModule,
    nn::loss::MseLoss,
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
    },
};
use clap::Args;
use serde::{Deserialize, Serialize};

impl<B: AutodiffBackend> TrainStep for PatchTST<B> {
    type Input = TimeSeriesBatch<B>;
    type Output = ForecastOutput<B>;
    fn step(&self, batch: TimeSeriesBatch<B>) -> TrainOutput<ForecastOutput<B>> {
        let TimeSeriesBatch {
            x,
            x_mark,
            y,
            y_mark,
        } = batch;
        let mut dec_input = Tensor::zeros_like(&y);
        dec_input = Tensor::cat(vec![y.clone(), dec_input], 1);
        let output = self.forecast(x, x_mark, dec_input, y_mark);
        let loss = MseLoss::new().forward(output.clone(), y.clone(), nn::loss::Reduction::Mean);
        let item = ForecastOutput::new(loss.clone(), output, y);
        TrainOutput::new(self, loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for PatchTST<B> {
    type Input = TimeSeriesBatch<B>;
    type Output = ForecastOutput<B>;

    fn step(&self, batch: TimeSeriesBatch<B>) -> ForecastOutput<B> {
        let TimeSeriesBatch {
            x,
            x_mark,
            y,
            y_mark,
        } = batch;
        let mut dec_input = Tensor::zeros_like(&y);
        dec_input = Tensor::cat(vec![y.clone(), dec_input], 1);
        let output = self.forecast(x, x_mark, dec_input, y_mark);
        let loss = MseLoss::new().forward(output.clone(), y.clone(), nn::loss::Reduction::Mean);
        ForecastOutput::new(loss.clone(), output, y)
    }
}

#[derive(Debug, Args, Clone, Deserialize, Serialize)]
pub struct TrainConfig {
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

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B, M>(
    artifact_dir: &str,
    train_config: TrainConfig,
    model: M,
    data_config: DataConfig,
    lengths: TimeLengths,
    device: B::Device,
) where
    B: AutodiffBackend,
    M: AutodiffModule<B>
        + TrainStep<Input = TimeSeriesBatch<B>>
        + InferenceStep
        + std::fmt::Display,
    <M as burn::module::AutodiffModule<B>>::InnerModule: burn::train::InferenceStep,
{
    create_artifact_dir(artifact_dir);
    train_config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

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

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(train_config.batch_size)
        .shuffle(train_config.seed)
        .num_workers(train_config.num_workers)
        .build(ETTHourDataset::new(
            &data_config,
            &lengths,
            ExpFlag::Train,
            &device,
        ));

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(train_config.num_epochs)
        .summary();
    let optimizer = AdamConfig::new().init();
    let result = training.launch(Learner::new(model, optimizer, train_config.learning_rate));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
