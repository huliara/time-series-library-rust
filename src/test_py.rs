use pyo3::prelude::*;
use std::env;

pub fn get_python_fnction(py: Python<'_>, name: String, attr_name: String) -> Bound<'_, PyAny> {
    let sys = py.import("sys").unwrap();

    // 1. Set up sys.path
    let current_dir = env::current_dir()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
        .unwrap();
    let current_dir_str = current_dir
        .to_str()
        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Invalid path"))
        .unwrap();

    let path = sys.getattr("path").unwrap();
    path.call_method1("append", (current_dir_str,)).unwrap();

    // 3. Import the module
    let module = py.import(name).unwrap();
    // 4. Get the function
    module.getattr(attr_name).unwrap()
}

pub fn execute_python_forward(model_name: &str) -> PyResult<Vec<f32>> {
    Python::attach(|py: Python<'_>| {
        let func = get_python_fnction(
            py,
            "_torch_forward_test".to_string(),
            "torch_forward_test".to_string(),
        );
        // 5. Call the function with model_name
        let result = func.call1((model_name,))?;

        // 6. Convert numpy result to flat Vec<f32>
        //    We expect the result to be a numpy array.
        //    Flattening ensures we get a 1D sequence.
        let flat_result = result.call_method0("flatten")?.call_method0("tolist")?;
        let output: Vec<f32> = flat_result.extract()?;

        Ok(output)
    })
}

pub fn execute_data_provider_test() -> PyResult<(Vec<f32>, Vec<f32>)> {
    Python::attach(|py| {
        let func = get_python_fnction(py, "_dataset_test".to_string(), "dataset_test".to_string());

        // 5. Call the function
        let result = func.call0()?;

        // 6. Extract tuple (x, y)
        let tuple_result = result.cast::<pyo3::types::PyTuple>()?;

        let x_val = tuple_result.get_item(0)?;
        let data_stamp = tuple_result.get_item(1)?;

        // 7. Convert to flat vectors
        let x_flat = x_val.call_method0("flatten")?.call_method0("tolist")?;
        let data_stamp_flat = data_stamp.call_method0("flatten")?.call_method0("tolist")?;

        let x_vec: Vec<f32> = x_flat.extract()?;
        let data_stamp_vec: Vec<f32> = data_stamp_flat.extract()?;

        Ok((x_vec, data_stamp_vec))
    })
}

pub fn execute_dataloader_test() -> PyResult<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    Python::attach(|py| {
        let func = get_python_fnction(
            py,
            "_dataloader_test".to_string(),
            "dataloader_test".to_string(),
        );

        // 5. Call the function
        let result = func.call0()?;

        // 6. Extract tuple (all_x, all_y, all_x_mark, all_y_mark)
        let tuple_result = result.cast::<pyo3::types::PyTuple>()?;

        let all_x = tuple_result.get_item(0)?;
        let all_y = tuple_result.get_item(1)?;
        let all_x_mark = tuple_result.get_item(2)?;
        let all_y_mark = tuple_result.get_item(3)?;
        let all_x_vec: Vec<f32> = all_x.extract()?;
        let all_y_vec: Vec<f32> = all_y.extract()?;
        let all_x_mark_vec: Vec<f32> = all_x_mark.extract()?;
        let all_y_mark_vec: Vec<f32> = all_y_mark.extract()?;
        println!("Received from Python - all_x length: {}, all_y length: {}, all_x_mark length: {}, all_y_mark length: {}", all_x_vec.len(), all_y_vec.len(), all_x_mark_vec.len(), all_y_mark_vec.len());

        Ok((all_x_vec, all_y_vec, all_x_mark_vec, all_y_mark_vec))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_python_forward() {
        let model_name = "PatchTST";
        let result: Result<Vec<f32>, PyErr> = execute_python_forward(model_name);
        if let Err(e) = &result {
            panic!("Python execution failed: {:?}", e);
        }
        let output = result.unwrap();
        // We expect some output; exact values depend on the Python implementation.
        assert!(!output.is_empty());
    }
}
