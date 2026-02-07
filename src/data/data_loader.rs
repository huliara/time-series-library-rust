use std::sync::Arc;

use crate::args::Args;

use crate::data::{
    batcher::{TimeSeriesBatch, TimeSeriesBatcher},
    dataset::ett_hour::{ETTHourDataset, ExpFlag},
};
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder},
    prelude::Backend,
};

pub fn create_data_loader<B: Backend>(
    args: &Args,
    flag: ExpFlag,
) -> Arc<dyn DataLoader<B, TimeSeriesBatch<B>>> {
    let device = B::Device::default();
    let dataset: ETTHourDataset<B> = ETTHourDataset::new(args, flag, &device);
    match flag {
        ExpFlag::Train => DataLoaderBuilder::new(TimeSeriesBatcher::default())
            .batch_size(args.batch_size)
            .shuffle(args.seed)
            .build(dataset),
        ExpFlag::Val => DataLoaderBuilder::new(TimeSeriesBatcher::default())
            .batch_size(args.batch_size)
            .build(dataset),
        ExpFlag::Test => DataLoaderBuilder::new(TimeSeriesBatcher::default())
            .batch_size(args.batch_size)
            .build(dataset),
    }
}

#[cfg(test)]
mod tests {
    use crate::test_py::execute_dataloader_test;
    use burn::backend::wgpu::Wgpu;
    use clap::Parser;

    use super::*;
    use crate::args::Args;
    #[test]
    fn test_create_dataloader() {
        type B = Wgpu;
        let args = Args::parse_from(vec![
            "test",
            "long-term-forecast",
            "single",
            "ot",
            "time-f",
            "wgpu",
            "--data-path",
            "data/ETT/ETTh1.csv",
            "--skip-training",
        ]);
        let data_loader = create_data_loader::<B>(&args, ExpFlag::Train);
        let py_dataloader_output =
            execute_dataloader_test().expect("Failed to execute dataloader test");

        let mut x_vec = Vec::with_capacity(3);
        let mut y_vec = Vec::with_capacity(3);
        let mut x_mark_vec = Vec::with_capacity(3);
        let mut y_mark_vec = Vec::with_capacity(3);
        for batch in data_loader.iter() {
            x_vec.push(batch.x);
            y_vec.push(batch.y);
            x_mark_vec.push(batch.x_mark);
            y_mark_vec.push(batch.y_mark);
        }
        let x_tensor = burn::tensor::Tensor::cat(x_vec, 0).to_data();
        let y_tensor = burn::tensor::Tensor::cat(y_vec, 0).to_data();
        let x_mark_tensor = burn::tensor::Tensor::cat(x_mark_vec, 0).to_data();
        let y_mark_tensor = burn::tensor::Tensor::cat(y_mark_vec, 0).to_data();

        let py_x_tensor =
            burn::tensor::TensorData::new(py_dataloader_output.0, x_tensor.clone().shape);
        let py_y_tensor =
            burn::tensor::TensorData::new(py_dataloader_output.1, y_tensor.clone().shape);
        let py_x_mark_tensor =
            burn::tensor::TensorData::new(py_dataloader_output.2, x_mark_tensor.clone().shape);
        let py_y_mark_tensor =
            burn::tensor::TensorData::new(py_dataloader_output.3, y_mark_tensor.clone().shape);
        assert_eq!(py_x_tensor.shape, x_tensor.shape);
        assert_eq!(py_y_tensor.shape, y_tensor.shape);
        assert_eq!(py_x_mark_tensor.shape, x_mark_tensor.shape);
        assert_eq!(py_y_mark_tensor.shape, y_mark_tensor.shape);
        py_x_tensor.assert_approx_eq::<f32>(&x_tensor, burn::tensor::Tolerance::default());
        py_y_tensor.assert_approx_eq::<f32>(&y_tensor, burn::tensor::Tolerance::default());
        py_x_mark_tensor
            .assert_approx_eq::<f32>(&x_mark_tensor, burn::tensor::Tolerance::default());
        py_y_mark_tensor
            .assert_approx_eq::<f32>(&y_mark_tensor, burn::tensor::Tolerance::default());
    }
}
