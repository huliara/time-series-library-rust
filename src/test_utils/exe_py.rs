use pyo3::prelude::*;
use std::env;

pub fn execute_python_forward(model_name: &str) -> PyResult<Vec<f32>> {
    Python::with_gil(|py| {
        let sys = py.import("sys")?;

        // 1. Set up sys.path
        let current_dir = env::current_dir()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let current_dir_str = current_dir
            .to_str()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Invalid path"))?;

        let path = sys.getattr("path")?;
        path.call_method1("append", (current_dir_str,))?;

        // 2. Mock sys.argv to avoid argparse errors when running from 'cargo test'
        //    which passes its own arguments.
        //    We pass just the script name, effectively simulating no arguments (default config).
        let argv = vec!["test_rust_model.py"];
        sys.setattr("argv", argv)?;

        // 3. Import the module
        let module = py.import("test_rust_model")?;

        // 4. Get the function
        let func = module.getattr("torch_forward_test")?;

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
    Python::with_gil(|py| {
        let sys = py.import("sys")?;

        // 1. Set up sys.path
        let current_dir = env::current_dir()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        let current_dir_str = current_dir
            .to_str()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Invalid path"))?;

        let path = sys.getattr("path")?;
        path.call_method1("append", (current_dir_str,))?;

        // 2. Mock sys.argv
        let argv = vec!["_data_provider_test.py"];
        sys.setattr("argv", argv)?;

        // 3. Import the module
        let module = py.import("_data_provider_test")?;

        // 4. Get the function
        let func = module.getattr("data_provider_test")?;

        // 5. Call the function
        let result = func.call0()?;

        // 6. Extract tuple (x, y)
        let tuple_result = result.downcast::<pyo3::types::PyTuple>()?;

        let x_val = tuple_result.get_item(0)?;
        let y_val = tuple_result.get_item(1)?;

        // 7. Convert to flat vectors
        let x_flat = x_val.call_method0("flatten")?.call_method0("tolist")?;
        let y_flat = y_val.call_method0("flatten")?.call_method0("tolist")?;

        let x_vec: Vec<f32> = x_flat.extract()?;
        let y_vec: Vec<f32> = y_flat.extract()?;

        Ok((x_vec, y_vec))
    })
}
