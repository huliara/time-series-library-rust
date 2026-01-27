use crate::data::test_utils::setup_test_dataloader;
use crate::models::forward::Forward;
use crate::test_py::execute_python_forward;
use burn::{
    tensor::{backend::Backend, TensorData},
    Tensor,
};
use std::any;
pub fn assert_module_forward<B: Backend, M: Forward<B>>(module: M) {
    let data_loader = setup_test_dataloader();
    let mut rust_vec = Vec::with_capacity(3);
    for batch in data_loader.iter() {
        let output = module.forward(batch.x, batch.x_mark, batch.y, batch.y_mark);
        rust_vec.push(output);
    }
    let rust_tensor = Tensor::cat(rust_vec, 0).to_data();

    let py_forward_results: Vec<f32> = execute_python_forward(any::type_name::<M>()).unwrap();

    let py_tensor = TensorData::new(py_forward_results, rust_tensor.clone().shape);

    assert_eq!(py_tensor.shape, rust_tensor.shape);

    py_tensor.assert_approx_eq::<f32>(&rust_tensor, burn::tensor::Tolerance::default());
}
