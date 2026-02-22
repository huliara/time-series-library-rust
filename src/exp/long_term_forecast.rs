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
        InferenceStep, Learner, LearningComponentsTypes, SupervisedTraining, TrainOutput,
        TrainStep, TrainingModel,
    },
};
use clap::Args;
use serde::{Deserialize, Serialize};

struct ForecastModel<B: Backend> {
    model: Model<B>,
}

impl<B: Backend> ForecastModel<B> {
    pub fn new(model_config: ModelConfig, device: &B::Device) -> Self {
        let model = match model_config {
            ModelConfig::PatchTST(args) => {
                Model::PatchTST(PatchTSTConfig::new(args).init(TaskName::LongTermForecast, device))
            }
            ModelConfig::DLinear(args) => {
                Model::DLinear(DLinearConfig::new(args).init(TaskName::LongTermForecast, device))
            }
        };
        ForecastModel { model }
    }
}

#[derive(Module, Debug)]
enum Model<B: Backend> {
    PatchTST(PatchTST<B>),
    DLinear(DLinear<B>),
}

impl<B: Backend> Forecast<B> for ForecastModel<B> {
    fn forecast(
        &self,
        x: Tensor<B, 3>,
        x_mark: Tensor<B, 3>,
        dec_input: Tensor<B, 3>,
        y_mark: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        match &self.model {
            Model::PatchTST(model) => model.forecast(x, x_mark, dec_input, y_mark),
            Model::DLinear(model) => model.forecast(x, x_mark, dec_input, y_mark),
        }
    }
}

impl<B: AutodiffBackend> TrainStep for ForecastModel<B> {
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
        TrainOutput::new(&self.model, loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for ForecastModel<B> {
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

pub fn train<B>(
    artifact_dir: &str,
    train_config: TrainConfig,
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
