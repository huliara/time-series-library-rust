use crate::{
    args::{data_config::DataConfig, model_config::ModelConfig, time_lengths::TimeLengths},
    data::{
        batcher::TimeSeriesBatcher,
        dataset::ett_hour::{ETTHourDataset, ExpFlag},
    },
    exp::{
        long_term_forecast::{save_results::save_results, train::ExpConfig, ForecastModel},
        Infer,
    },
    models::traits::Forecast,
};
use burn::{
    data::dataloader::DataLoaderBuilder,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
};

impl<B: AutodiffBackend> Infer<B> for ForecastModel<B> {
    fn infer(
        &self,
        exp_root_path: &str,
        exp_config: ExpConfig,
        model_config: ModelConfig,
        lengths: TimeLengths,
        data_config: DataConfig,
        device: B::Device,
    ) {
        let batcher = TimeSeriesBatcher::default();
        let record = CompactRecorder::new()
            .load(format!("{exp_root_path}/model").into(), &device)
            .expect("Trained model should exist; run train first");

        let model: ForecastModel<B> =
            ForecastModel::<B>::new(model_config, &device).load_record(record);
        let dataloader_test = DataLoaderBuilder::new(batcher)
            .batch_size(exp_config.batch_size)
            .shuffle(exp_config.seed)
            .num_workers(exp_config.num_workers)
            .build(ETTHourDataset::new(
                &data_config,
                &lengths,
                ExpFlag::Test,
                &device,
            ));
        let mut _predicts = Vec::with_capacity(3);
        let mut _futures = Vec::with_capacity(3);
        for batch in dataloader_test.iter() {
            let output = model.forecast(batch.x, batch.x_mark, batch.y.clone(), batch.y_mark);
            _predicts.push(output);
            _futures.push(batch.y);
        }
        let predicts = Tensor::cat(_predicts, 0);
        let futures = Tensor::cat(_futures, 0);
        let error = predicts.clone() - futures.clone();
        save_results(exp_root_path, error, predicts, futures);
    }
}
