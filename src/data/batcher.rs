use crate::data::dataset::ett_hour::TimeSeriesItem;
use burn::{data::dataloader::batcher::Batcher, prelude::Backend, tensor::Tensor};

#[derive(Clone, Debug, Default)]
pub struct TimeSeriesBatcher {}

#[derive(Clone, Debug)]
pub struct TimeSeriesBatch<B: Backend> {
    pub x: Tensor<B, 3>,
    pub x_mark: Tensor<B, 3>,
    pub y: Tensor<B, 3>,
    pub y_mark: Tensor<B, 3>,
}

impl<B: Backend> Batcher<B, TimeSeriesItem<B>, TimeSeriesBatch<B>> for TimeSeriesBatcher {
    fn batch(&self, items: Vec<TimeSeriesItem<B>>, device: &B::Device) -> TimeSeriesBatch<B> {
        let x = Tensor::stack(items.iter().map(|item| item.seq_x.clone()).collect(), 0)
            .to_device(device);
        let x_mark = Tensor::stack(
            items.iter().map(|item| item.seq_x_mark.clone()).collect(),
            0,
        )
        .to_device(device);
        let y = Tensor::stack(items.iter().map(|item| item.seq_y.clone()).collect(), 0)
            .to_device(device);
        let y_mark = Tensor::stack(
            items.iter().map(|item| item.seq_y_mark.clone()).collect(),
            0,
        )
        .to_device(device);

        TimeSeriesBatch {
            x,
            x_mark,
            y,
            y_mark,
        }
    }
}
