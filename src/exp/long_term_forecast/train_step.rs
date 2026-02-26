use crate::{
    data::batcher::TimeSeriesBatch,
    exp::long_term_forecast::{forecast_output::ForecastOutput, ForecastModel},
    models::traits::Forecast,
};
use burn::{
    nn::loss::MseLoss,
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{TrainOutput, TrainStep},
};

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
