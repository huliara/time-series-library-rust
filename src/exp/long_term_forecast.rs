mod forecast_output;
pub mod infer;
mod infer_step;
mod save_results;
pub mod train;
mod train_step;
use crate::{
    args::{exp::TaskName, model_config::ModelConfig, time_lengths::TimeLengths},
    exp::Exp,
    models::{
        dlinear::{DLinear, DLinearConfig},
        patch_tst::{PatchTST, PatchTSTConfig},
        traits::Forecast,
    },
};
use burn::{prelude::*, tensor::backend::AutodiffBackend};
#[derive(Module, Debug)]
enum Model<B: Backend> {
    PatchTST(PatchTST<B>),
    DLinear(DLinear<B>),
}

#[derive(Module, Debug)]
pub struct ForecastModel<B: Backend> {
    model: Model<B>,
}

impl<B: Backend> ForecastModel<B> {
    pub fn new(model_config: ModelConfig, lengths: TimeLengths, device: &B::Device) -> Self {
        let model = match model_config {
            ModelConfig::PatchTST(args) => Model::PatchTST(PatchTSTConfig::new(args).init(
                TaskName::LongTermForecast,
                lengths,
                device,
            )),
            ModelConfig::DLinear(args) => Model::DLinear(DLinearConfig::new(args).init(
                TaskName::LongTermForecast,
                lengths,
                device,
            )),
        };
        ForecastModel { model }
    }
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

impl<B: AutodiffBackend> Exp<B> for ForecastModel<B> {}
