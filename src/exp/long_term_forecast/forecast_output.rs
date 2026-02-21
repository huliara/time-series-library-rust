use burn::{
    prelude::Backend,
    tensor::Transaction,
    train::{
        metric::{Adaptor, LossInput},
        ItemLazy,
    },
    Tensor,
};
use burn_ndarray::NdArray;

pub struct ForecastOutput<B: Backend> {
    pub loss: Tensor<B, 1>,
    pub output: Tensor<B, 3>,
    pub targets: Tensor<B, 3>,
}

impl<B: Backend> ForecastOutput<B> {
    pub fn new(loss: Tensor<B, 1>, output: Tensor<B, 3>, targets: Tensor<B, 3>) -> Self {
        Self {
            loss,
            output,
            targets,
        }
    }
}

impl<B: Backend> Adaptor<LossInput<B>> for ForecastOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: Backend> ItemLazy for ForecastOutput<B> {
    type ItemSync = ForecastOutput<NdArray>;

    fn sync(self) -> Self::ItemSync {
        let [output, loss, targets] = Transaction::default()
            .register(self.output)
            .register(self.loss)
            .register(self.targets)
            .execute()
            .try_into()
            .expect("Correct amount of tensor data");

        let device = &Default::default();

        ForecastOutput {
            output: Tensor::from_data(output, device),
            loss: Tensor::from_data(loss, device),
            targets: Tensor::from_data(targets, device),
        }
    }
}
