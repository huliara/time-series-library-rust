mod forecast_output;
pub mod infer;
mod infer_step;
pub mod train;
mod train_step;
use crate::{
    args::{exp::TaskName, model_config::ModelConfig},
    models::{
        dlinear::{DLinear, DLinearConfig},
        patch_tst::{PatchTST, PatchTSTConfig},
        traits::Forecast,
    },
};
use burn::prelude::*;

#[derive(Module, Debug)]
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
