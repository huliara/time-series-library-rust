pub mod ett_hour;
pub mod util;

use burn::prelude::Backend;

use crate::{
    args::{
        data_config::{Data, DataConfig},
        time_lengths::TimeLengths,
    },
    data::dataset::ett_hour::{ETTHourDataset, ExpFlag},
};

pub fn get_dataset<B: Backend>(
    data_config: &DataConfig,
    lengths: &TimeLengths,
    flag: ExpFlag,
    device: &B::Device,
) -> ETTHourDataset<B> {
    match data_config.data {
        Data::ETTh1 => ETTHourDataset::new(data_config, lengths, flag, device),
    }
}
