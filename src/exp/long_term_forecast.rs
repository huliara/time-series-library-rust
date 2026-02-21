mod forecast_output;
use crate::{
    args::{RootArgs, exp::TaskName, model_config::ModelConfig},
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

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device) {
    create_artifact_dir(artifact_dir);
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(&device, config.seed);

    let batcher = TimeSeriesBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ETTHourDataset::new(
            &RootArgs::default().data_config,
            &RootArgs::default().lengths,
            ExpFlag::_Train,
            &device,
        ));

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::);

    let training = SupervisedTraining::new(artifact_dir, dataloader_train, dataloader_test)
        .metrics((AccuracyMetric::new(), LossMetric::new()))
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let model = config.model.init::<B>(&device);
    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}
