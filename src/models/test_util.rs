use crate::data::test_utils::setup_test_dataloader;
use crate::models::traits::Forecast;
use crate::test_py::execute_python_forward;
use burn::{
    tensor::{backend::Backend, TensorData},
    Tensor,
};
use std::any::type_name;
pub fn assert_module_forecast<B: Backend, M: Forecast<B>>(module: M) {
    let data_loader = setup_test_dataloader();
    let mut rust_vec = Vec::with_capacity(3);
    for batch in data_loader.iter() {
        let output = module.forecast(batch.x, batch.x_mark, batch.y, batch.y_mark);
        rust_vec.push(output);
    }
    let rust_tensor = Tensor::cat(rust_vec, 0).to_data();
    let type_name_vec: Vec<&str> = type_name::<M>().split('<').collect();
    let type_name = type_name_vec[0].split("::").last().unwrap();
    let py_forward_results: Vec<f32> = execute_python_forward(type_name).unwrap();

    let py_tensor = TensorData::new(py_forward_results, rust_tensor.clone().shape);

    assert_eq!(py_tensor.shape, rust_tensor.shape);

    py_tensor.assert_approx_eq::<f32>(&rust_tensor, burn::tensor::Tolerance::default());
}
