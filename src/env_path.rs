use crate::args::data_config::Data;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct EnvPath {
    pub result_root_path: String,
    pub data_root_path: String,
    pub dataset_path: HashMap<String, String>,
}

fn get_env_paths() -> EnvPath {
    let file = File::open("env.yml").expect("env.yml not found in the root directory");
    let reader = BufReader::new(file);
    serde_yaml::from_reader(reader).expect("Failed to parse env.yml")
}

pub fn get_result_root_path() -> String {
    get_env_paths().result_root_path
}

pub fn get_dataset_path(data: Data) -> String {
    let paths = get_env_paths();
    let key = data.to_string();
    let relative_path = paths
        .dataset_path
        .get(&key)
        .cloned()
        .unwrap_or_else(|| panic!("Dataset path for {} not found in env.yml", key));

    Path::new(&paths.data_root_path)
        .join(relative_path)
        .to_str()
        .unwrap()
        .to_string()
}
