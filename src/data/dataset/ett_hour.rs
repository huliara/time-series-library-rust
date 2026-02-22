use super::util::*;
use crate::args::{
    data_config::DataConfig, feature_type::FeatureType, time_embed::TimeEmbed,
    time_lengths::TimeLengths,
};
use burn::{
    data::dataset::Dataset,
    tensor::{backend::Backend, Tensor, TensorData},
};
use ndarray::{s, Array1, Array2, Axis};
use polars::prelude::*;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct TimeSeriesItem<B: Backend> {
    pub seq_x: Tensor<B, 2>,
    pub seq_y: Tensor<B, 2>,
    pub seq_x_mark: Tensor<B, 2>,
    pub seq_y_mark: Tensor<B, 2>,
}

#[derive(Clone, Debug)]
pub struct StandardScaler {
    pub mean: Array1<f64>,
    pub scale: Array1<f64>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            mean: Array1::zeros(0),
            scale: Array1::zeros(0),
        }
    }

    pub fn fit(&mut self, data: &Array2<f64>) {
        self.mean = data.mean_axis(Axis(0)).expect("Mean axis 0 failed");
        // Using ddof=0 for consistency with sklearn's StandardScaler which uses biased estimator by default
        self.scale = data.std_axis(Axis(0), 0.0);
        // Avoid division by zero
        self.scale.mapv_inplace(|x| if x == 0.0 { 1.0 } else { x });
    }

    pub fn transform(&self, data: &Array2<f64>) -> Array2<f64> {
        (data - &self.mean) / &self.scale
    }

    pub fn _inverse_transform(&self, data: &Array2<f64>) -> Array2<f64> {
        (data * &self.scale) + &self.mean
    }
}

pub struct ETTHourDataset<B: Backend> {
    pub data_x: Tensor<B, 2>,
    pub data_y: Tensor<B, 2>,
    pub data_stamp: Tensor<B, 2>,
    pub seq_len: usize,
    pub label_len: usize,
    pub pred_len: usize,
}

#[derive(Clone, Copy, Debug)]
pub enum ExpFlag {
    Train,
    _Val,
    Test,
}

impl<B: Backend> ETTHourDataset<B> {
    pub fn new(
        args: &DataConfig,
        lengths: &TimeLengths,
        flag: ExpFlag,
        device: &B::Device,
    ) -> Self {
        // Default size

        let path = PathBuf::from(&args.data_path);
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(path))
            .expect("Failed to read CSV file")
            .finish();

        match df {
            Ok(df) => {
                let border1s = (
                    0,
                    12 * 30 * 24 - lengths.seq_len,
                    12 * 30 * 24 + 4 * 30 * 24 - lengths.seq_len,
                );
                let border2s: (usize, usize, usize) = (
                    12 * 30 * 24,
                    12 * 30 * 24 + 4 * 30 * 24,
                    12 * 30 * 24 + 8 * 30 * 24,
                );

                let (start_idx, end_idx) = match flag {
                    ExpFlag::Train => (border1s.0, border2s.0),
                    ExpFlag::_Val => (border1s.1, border2s.1),
                    ExpFlag::Test => (border1s.2, border2s.2),
                };

                let data_array: Array2<f64> = match args.feature_type {
                    FeatureType::Multi => df
                        .clone()
                        .lazy()
                        .select([
                            col("HUFL"),
                            col("HULL"),
                            col("MUFL"),
                            col("MULL"),
                            col("LUFL"),
                            col("LULL"),
                            col("OT"),
                        ])
                        .collect()
                        .unwrap()
                        .to_ndarray::<Float64Type>(IndexOrder::C)
                        .unwrap()
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap(),
                    FeatureType::Single => df
                        .clone()
                        .lazy()
                        .select([col(args.target.to_string())])
                        .collect()
                        .unwrap()
                        .to_ndarray::<Float64Type>(IndexOrder::C)
                        .unwrap()
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap(),
                };

                let mut scaler = StandardScaler::new();
                let train_data = data_array.slice(s![border1s.0..border2s.0, ..]).to_owned();
                scaler.fit(&train_data);
                let data = scaler.transform(&data_array);

                let slice_len = end_idx - start_idx;

                let data_stamp_array: Array2<f32> = match args.embed {
                    TimeEmbed::Fixed => {
                        use chrono::{Datelike, Timelike};
                        let dates: Vec<chrono::NaiveDateTime> = df
                            .slice(start_idx as i64, slice_len)
                            .column("date")
                            .unwrap()
                            .str()
                            .unwrap()
                            .into_no_null_iter()
                            .map(|s| {
                                chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
                                    .expect("Parse date")
                            })
                            .collect();

                        let month: Vec<f32> = dates.iter().map(|d| d.month() as f32).collect();
                        let day: Vec<f32> = dates.iter().map(|d| d.day() as f32).collect();
                        let weekday: Vec<f32> = dates
                            .iter()
                            .map(|d| d.weekday().number_from_monday() as f32 - 1.0)
                            .collect();
                        let hour: Vec<f32> = dates.iter().map(|d| d.hour() as f32).collect();

                        let month_series = Column::new("month".into(), month);
                        let day_series = Column::new("day".into(), day);
                        let weekday_series = Column::new("weekday".into(), weekday);
                        let hour_series = Column::new("hour".into(), hour);

                        DataFrame::new(vec![month_series, day_series, weekday_series, hour_series])
                            .unwrap()
                            .to_ndarray::<Float32Type>(IndexOrder::C)
                            .unwrap()
                            .into_dimensionality::<ndarray::Ix2>()
                            .unwrap()
                    }
                    TimeEmbed::TimeF => {
                        let dates: Vec<chrono::NaiveDateTime> = df
                            .slice(start_idx as i64, slice_len)
                            .column("date")
                            .unwrap()
                            .str()
                            .unwrap()
                            .into_no_null_iter()
                            .map(|s| {
                                chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
                                    .expect("Parse date")
                            })
                            .collect();
                        time_features(&dates, "h")
                    }
                };

                let data_x_array = data.slice(s![start_idx..end_idx, ..]).to_owned();
                let data_y_array = data.slice(s![start_idx..end_idx, ..]).to_owned();

                let shape_x = data_x_array.shape().to_vec();
                let data_x = Tensor::from_data(
                    TensorData::new(data_x_array.into_raw_vec_and_offset().0, shape_x),
                    device,
                );

                let shape_y = data_y_array.shape().to_vec();
                let data_y = Tensor::from_data(
                    TensorData::new(data_y_array.into_raw_vec_and_offset().0, shape_y),
                    device,
                );

                let shape_stamp = data_stamp_array.shape().to_vec();
                let data_stamp = Tensor::from_data(
                    TensorData::new(data_stamp_array.into_raw_vec_and_offset().0, shape_stamp),
                    device,
                );

                Self {
                    data_x,
                    data_y,
                    data_stamp,
                    seq_len: lengths.seq_len,
                    label_len: lengths.label_len,
                    pred_len: lengths.pred_len,
                }
            }
            Err(e) => {
                panic!("Error reading CSV file: {:?}", e);
            }
        }
    }
}

impl<B: Backend> Dataset<TimeSeriesItem<B>> for ETTHourDataset<B> {
    fn get(&self, index: usize) -> Option<TimeSeriesItem<B>> {
        if index >= self.len() {
            return None;
        }
        let s_begin = index;
        let s_end = s_begin + self.seq_len;
        let r_begin = s_end - self.label_len;
        let r_end = r_begin + self.label_len + self.pred_len;

        let dim_x = self.data_x.dims()[1];
        let dim_mark = self.data_stamp.dims()[1];

        // Slicing in Burn: ranges for each dimension
        let seq_x = self.data_x.clone().slice([s_begin..s_end, 0..dim_x]);
        let seq_y = self.data_y.clone().slice([r_begin..r_end, 0..dim_x]);
        let seq_x_mark = self.data_stamp.clone().slice([s_begin..s_end, 0..dim_mark]);
        let seq_y_mark = self.data_stamp.clone().slice([r_begin..r_end, 0..dim_mark]);

        Some(TimeSeriesItem {
            seq_x,
            seq_y,
            seq_x_mark,
            seq_y_mark,
        })
    }

    fn len(&self) -> usize {
        let len_x = self.data_x.dims()[0];
        let required = self.seq_len + self.pred_len;
        if len_x >= required {
            len_x - required + 1
        } else {
            0
        }
    }
}
#[cfg(test)]
mod tests {
    use super::ETTHourDataset;
    use crate::args::data_config::DataConfig;
    use crate::args::time_lengths::TimeLengths;
    use crate::test_py::execute_data_provider_test;
    use burn::{tensor::TensorData, tensor::Tolerance};
    #[test]
    fn test_ett_hour_dataset() {
        type B = burn::backend::wgpu::Wgpu;
        let py_dataset_result = execute_data_provider_test().unwrap();
        let device = Default::default();
        let data_config = DataConfig::default();
        let lengths = TimeLengths::default();
        let rust_dataset =
            ETTHourDataset::<B>::new(&data_config, &lengths, super::ExpFlag::Test, &device);

        let py_tensor_stamp = TensorData::new(py_dataset_result.1, rust_dataset.data_stamp.shape());

        let rust_tensor_stamp = rust_dataset.data_stamp.to_data();
        assert_eq!(py_tensor_stamp.shape, rust_tensor_stamp.shape);
        py_tensor_stamp.assert_approx_eq::<f32>(&rust_tensor_stamp, Tolerance::default());
        let py_tensor_x = TensorData::new(py_dataset_result.0, rust_dataset.data_x.shape());
        let rust_tensor_x = rust_dataset.data_x.to_data();
        py_tensor_x.assert_approx_eq::<f32>(&rust_tensor_x, Tolerance::default());
        assert_eq!(py_tensor_x.shape, rust_tensor_x.shape);
    }
}
