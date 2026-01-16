use pyo3::prelude::*;
use pyo3::types::PyModule;
use std::collections::HashMap;

const PYTHON_SCRIPT: &str = r#"
import torch
import sys
import argparse
import numpy as np
import traceback

# Add repository root to path
sys.path.append('.')

def run_forward(model_name, config_dict, input_list, input_shape):
    try:
        # Create config namespace from dictionary
        configs = argparse.Namespace(**config_dict)
        
        # Set default values if missing (based on typical defaults in run.py)
        defaults = {
            'task_name': 'long_term_forecast',
            'pred_len': 96,
            'label_len': 48,
            'output_attention': False,
            'enc_in': 7,
            'dec_in': 7,
            'c_out': 7,
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 2048,
            'moving_avg': 25,
            'factor': 1,
            'distil': True,
            'dropout': 0.1,
            'embed': 'timeF',
            'activation': 'gelu',
            'num_class': 1,
            'seq_len': 96,
        }
        
        for k, v in defaults.items():
            if not hasattr(configs, k):
                setattr(configs, k, v)

        # Dynamic import of the model
        # Try finding it in 'models' first, then 'layers'
        try:
            module = __import__(f'models.{model_name}', fromlist=['Model'])
        except ImportError:
            try:
                module = __import__(f'layers.{model_name}', fromlist=['Model'])
            except ImportError as e:
                return f"Error: Could not import model {model_name} from models or layers. Details: {str(e)}"
            
        if not hasattr(module, 'Model'):
             return f"Error: Module {model_name} does not have a Model class"

        Model = getattr(module, 'Model')
        # Some models might fail verification if configs are missing crucial args
        try:
            model = Model(configs).float()
        except Exception as e:
             return f"Error building model: {str(e)}\n{traceback.format_exc()}"

        model.eval()

        # Prepare input tensor
        try:
            input_tensor = torch.tensor(input_list).reshape(input_shape).float()
        except Exception as e:
            return f"Error reshaping input tensor: {str(e)}"
            
        # Prepare dummy auxiliary inputs based on task
        B = input_tensor.shape[0]
        L = input_tensor.shape[1] # seq_len
        C = input_tensor.shape[2] # enc_in
        
        pred_len = configs.pred_len
        label_len = configs.label_len
        
        # x_mark_enc: [Batch, Seq_Len, 4] (assuming 4 time features)
        x_mark_enc = torch.zeros(B, L, 4).float()
        
        # x_dec: [Batch, Label_Len + Pred_Len, Dec_In]
        dec_in = configs.dec_in
        x_dec = torch.zeros(B, label_len + pred_len, dec_in).float()
        
        # x_mark_dec: [Batch, Label_Len + Pred_Len, 4]
        x_mark_dec = torch.zeros(B, label_len + pred_len, 4).float()

        # Forward pass
        with torch.no_grad():
            try:
                if configs.task_name == 'long_term_forecast' or configs.task_name == 'short_term_forecast':
                    outputs = model(input_tensor, x_mark_enc, x_dec, x_mark_dec)
                else:
                    outputs = model(input_tensor, x_mark_enc, x_dec, x_mark_dec)
            except TypeError:
                outputs = model(input_tensor)

            # Handle tuple outputs (e.g., output, attention)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

        return outputs.detach().cpu().numpy().flatten().tolist()
    
    except Exception as e:
        return f"Error: {str(e)}\n{traceback.format_exc()}"

"#;

pub fn forward_pytorch_model(
    model_name: &str,
    input_data: Vec<f32>,
    input_shape: Vec<usize>,
    config_map: HashMap<String, String>,
) -> PyResult<Vec<f32>> {
    Python::with_gil(|py| {
        let activators = PyModule::from_code(py, PYTHON_SCRIPT, "model_runner.py", "model_runner")?;

        // Convert config_map to PyDict
        let py_config = pyo3::types::PyDict::new(py);
        for (k, v) in config_map {
            if let Ok(i) = v.parse::<i32>() {
                py_config.set_item(k, i)?;
            } else if let Ok(f) = v.parse::<f64>() {
                py_config.set_item(k, f)?;
            } else if v == "True" || v == "true" {
                py_config.set_item(k, true)?;
            } else if v == "False" || v == "false" {
                py_config.set_item(k, false)?;
            } else {
                py_config.set_item(k, v)?;
            }
        }

        let args = (model_name, py_config, input_data, input_shape);
        let result = activators.getattr("run_forward")?.call1(args)?;

        // Check for error string
        if let Ok(error_msg) = result.extract::<String>() {
            if error_msg.starts_with("Error:") {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(error_msg));
            }
        }

        // Extract result
        let output_vec: Vec<f32> = result.extract()?;
        Ok(output_vec)
    })
}
