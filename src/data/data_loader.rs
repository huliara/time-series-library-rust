use std::sync::Arc;

use crate::args::{data_config::DataConfig, time_lengths::TimeLengths};

use crate::data::dataset::get_dataset;
use crate::data::{
    batcher::{TimeSeriesBatch, TimeSeriesBatcher},
    dataset::ett_hour::{ETTHourDataset, ExpFlag},
};
use burn::{
    data::dataloader::{DataLoader, DataLoaderBuilder},
    prelude::Backend,
};

pub fn create_data_loader<B: Backend>(
    data_config: &DataConfig,
    lengths: &TimeLengths,
    batch_size: usize,
    num_workers: usize,
    seed: u64,
    flag: ExpFlag,
) -> Arc<dyn DataLoader<B, TimeSeriesBatch<B>>> {
    let device = B::Device::default();
    let dataset: ETTHourDataset<B> = get_dataset(data_config, lengths, flag, &device);
    match flag {
        ExpFlag::Train | ExpFlag::Val => DataLoaderBuilder::new(TimeSeriesBatcher::default())
            .batch_size(batch_size)
            .shuffle(seed)
            .num_workers(num_workers)
            .build(dataset),

        ExpFlag::Test => DataLoaderBuilder::new(TimeSeriesBatcher::default())
            .batch_size(batch_size)
            .num_workers(num_workers)
            .build(dataset),
    }
}

#[cfg(test)]
mod tests {
    use crate::test_py::execute_dataloader_test;
    use burn::backend::wgpu::Wgpu;

    use super::*;

    #[test]
    fn test_create_dataloader() {
        type B = Wgpu;
        let data_config = DataConfig::default();
        let lengths = TimeLengths::default();
        let batch_size = 32;
        let num_workers = 0;
        let seed = 42;
        let data_loader = create_data_loader::<B>(
            &data_config,
            &lengths,
            batch_size,
            num_workers,
            seed,
            ExpFlag::Test,
        );
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
        println!("Rust x_tensor shape: {:?}\n", x_tensor.shape);
        println!("Rust y_tensor shape: {:?}\n", y_tensor.shape);
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
