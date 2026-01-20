use super::util::*;
use crate::args::{Args, FeatureType, Target, TimeEmbed};
use burn::data::dataset::{transform::WindowsDataset, Dataset, InMemDataset};
use burn::train::train;
use csv::ReaderBuilder;
use ndarray::{s, Array1, Array2, Axis};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct ETTHourItem {
    pub seq_x: Array2<f32>,
    pub seq_y: Array2<f32>,
    pub seq_x_mark: Array2<f32>,
    pub seq_y_mark: Array2<f32>,
}

#[derive(Clone, Debug)]
pub struct StandardScaler {
    pub mean: Array1<f32>,
    pub scale: Array1<f32>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            mean: Array1::zeros(0),
            scale: Array1::zeros(0),
        }
    }

    pub fn fit(&mut self, data: &Array2<f32>) {
        self.mean = data.mean_axis(Axis(0)).expect("Mean axis 0 failed");
        // Using ddof=0 for consistency with sklearn's StandardScaler which uses biased estimator by default
        self.scale = data.std_axis(Axis(0), 0.0);
        // Avoid division by zero
        self.scale.mapv_inplace(|x| if x == 0.0 { 1.0 } else { x });
    }

    pub fn transform(&self, data: &Array2<f32>) -> Array2<f32> {
        (data - &self.mean) / &self.scale
    }

    pub fn inverse_transform(&self, data: &Array2<f32>) -> Array2<f32> {
        (data * &self.scale) + &self.mean
    }
}

pub struct ETTHourDataset {
    pub data_x: Array2<f32>,
    pub data_y: Array2<f32>,
    pub data_stamp: Array2<f32>,
    pub seq_len: usize,
    pub label_len: usize,
    pub pred_len: usize,
    pub scaler: StandardScaler,
}

pub enum ExpFlag {
    Train,
    Val,
    Test,
}

impl ETTHourDataset {
    pub fn new(args: &Args, flag: ExpFlag) -> Self {
        // Default size
        let seq_len = args.seq_len;
        let label_len = args.label_len;
        let pred_len = args.pred_len;
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .try_into_reader_with_file_path(Some(args.data_path.into()))
            .expect("Failed to read CSV file")
            .finish();

        match df {
            Ok(df) => {
                let border1s = (
                    0,
                    12 * 30 * 24 - seq_len,
                    12 * 30 * 24 + 4 * 30 * 24 - seq_len,
                );
                let border2s: (usize, usize, usize) = (
                    12 * 30 * 24,
                    12 * 30 * 24 + 4 * 30 * 24,
                    12 * 30 * 24 + 8 * 30 * 24,
                );

                let (start_idx, end_idx) = match flag {
                    ExpFlag::Train => (border1s.0, border2s.0),
                    ExpFlag::Val => (border1s.1, border2s.1),
                    ExpFlag::Test => (border1s.2, border2s.2),
                };

                let feature_columns = match args.feature_type {
                    FeatureType::Multi => vec![
                        col("HUFL"),
                        col("HULL"),
                        col("MUFL"),
                        col("MULL"),
                        col("LUFL"),
                        col("LULL"),
                        col("OT"),
                    ],
                    FeatureType::Single => vec![col(&args.target.to_string())],
                };

                let data_array: Array2<f32> = df
                    .select(feature_columns)
                    .unwrap()
                    .to_ndarray::<Float32Type>()
                    .unwrap()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();

                let mut scaler = StandardScaler::new();
                let train_data = data_array.slice(s![border1s.0..border2s.0, ..]).to_owned();
                scaler.fit(&train_data);
                let data = scaler.transform(&data_array);

                let slice_len = (end_idx - start_idx) as usize;

                let date_series = df
                    .slice(start_idx as i64, slice_len)
                    .column("date")
                    .unwrap()
                    .str()
                    .unwrap()
                    .to_datetime(
                        TimeUnit::Microseconds,
                        None,
                        StrptimeOptions {
                            format: Some("%Y-%m-%d %H:%M:%S".into()),
                            strict: false,
                            exact: true,
                            ..Default::default()
                        },
                        "raise",
                    )
                    .unwrap();
                let data_stamp = match args.embed {
                    TimeEmbed::TimeF => {
                        let month = date_series
                            .month()
                            .into_series()
                            .cast(&DataType::Float32)
                            .unwrap();
                        let day = date_series
                            .day()
                            .into_series()
                            .cast(&DataType::Float32)
                            .unwrap();
                        let weekday = (date_series
                            .weekday()
                            .into_series()
                            .cast(&DataType::Float32)
                            .unwrap()
                            - 1.0);
                        let hour = date_series
                            .hour()
                            .into_series()
                            .cast(&DataType::Float32)
                            .unwrap();

                        DataFrame::new(vec![month, day, weekday, hour])
                            .unwrap()
                            .to_ndarray::<Float32Type>()
                            .unwrap()
                            .into_dimensionality::<ndarray::Ix2>()
                            .unwrap()
                    }
                    TimeEmbed::Fixed => {
                        let dates: Vec<chrono::NaiveDateTime> = date_series
                            .utf8()
                            .unwrap()
                            .into_no_null_iter()
                            .map(|s| {
                                chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
                                    .unwrap()
                            })
                            .collect();
                        time_features(&dates, "h")
                    }
                };

                let data_x = data.slice(s![start_idx..end_idx, ..]).to_owned();
                let data_y = data.slice(s![start_idx..end_idx, ..]).to_owned();

                Self {
                    data_x,
                    data_y,
                    data_stamp,
                    seq_len,
                    label_len,
                    pred_len,
                    scaler,
                }
            }
            Err(e) => {
                panic!("Error reading CSV file: {:?}", e);
            }
        }
    }

    pub fn inverse_transform(&self, data: &Array2<f32>) -> Array2<f32> {
        self.scaler.inverse_transform(data)
    }
}

impl Dataset<ETTHourItem> for ETTHourDataset {
    fn get(&self, index: usize) -> Option<ETTHourItem> {
        if index >= self.len() {
            return None;
        }
        let s_begin = index;
        let s_end = s_begin + self.seq_len;
        let r_begin = s_end - self.label_len;
        let r_end = r_begin + self.label_len + self.pred_len;

        let seq_x = self.data_x.slice(s![s_begin..s_end, ..]).to_owned();
        let seq_y = self.data_y.slice(s![r_begin..r_end, ..]).to_owned();
        let seq_x_mark = self.data_stamp.slice(s![s_begin..s_end, ..]).to_owned();
        let seq_y_mark = self.data_stamp.slice(s![r_begin..r_end, ..]).to_owned();

        Some(ETTHourItem {
            seq_x,
            seq_y,
            seq_x_mark,
            seq_y_mark,
        })
    }

    fn len(&self) -> usize {
        let len_x = self.data_x.len_of(Axis(0));
        let required = self.seq_len + self.pred_len;
        if len_x >= required {
            len_x - required + 1
        } else {
            0
        }
    }
}
