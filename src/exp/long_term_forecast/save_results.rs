use burn::{prelude::Backend, Tensor};
use plotters::prelude::*;
use std::fs;
use std::io::Write;

pub fn save_results<B: Backend>(
    exp_root_path: &str,
    error: Tensor<B, 3>,
    predicts: Tensor<B, 3>,
    futures: Tensor<B, 3>,
) {
    // Calculate MSE and MAE per time step (average over Batch and Features)
    // error shape: [Batch, Time, Features]
    // mean_dim(0) -> [1, Time, Features]
    // mean_dim(2) -> [1, Time, 1]
    // squeeze -> [Time]
    let mse_t = error
        .clone()
        .powf_scalar(2.0)
        .mean_dims(&[0, 2])
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let mae_t = error
        .clone()
        .abs()
        .mean_dims(&[0, 2])
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    fs::create_dir_all(format!("{exp_root_path}/test/")).unwrap();
    let mut mse_writer = csv::Writer::from_path(format!("{exp_root_path}/test/mse.csv")).unwrap();
    let mut mae_writer = csv::Writer::from_path(format!("{exp_root_path}/test/mae.csv")).unwrap();

    for val in mse_t.iter() {
        mse_writer.write_record(&[val.to_string()]).unwrap();
    }
    for val in mae_t.iter() {
        mae_writer.write_record(&[val.to_string()]).unwrap();
    }
    mse_writer.flush().unwrap();
    mae_writer.flush().unwrap();

    let all_mse = error
        .clone()
        .powf_scalar(2.0)
        .mean()
        .into_data()
        .into_vec::<f32>()
        .unwrap()[0];
    let all_mae = error
        .clone()
        .abs()
        .mean()
        .into_data()
        .into_vec::<f32>()
        .unwrap()[0];
    let all_rmse = error
        .clone()
        .powf_scalar(2.0)
        .sqrt()
        .mean()
        .into_data()
        .into_vec::<f32>()
        .unwrap()[0];
    let all_mape = (error.clone() / futures.clone())
        .abs()
        .mean()
        .into_data()
        .into_vec::<f32>()
        .unwrap()[0];
    let all_mspe = (error.clone() / futures.clone())
        .powf_scalar(2.0)
        .mean()
        .into_data()
        .into_vec::<f32>()
        .unwrap()[0];
    let mut file = std::fs::File::create(format!("{exp_root_path}/test/results.txt")).unwrap();
    file.write_all(
        format!(
            "MSE: {all_mse}\nMAE: {all_mae}\nRMSE: {all_rmse}\nMAPE: {all_mape}\nMSPE: {all_mspe}"
        )
        .as_bytes(),
    )
    .unwrap();

    // Plot 10 samples
    let batch_size = predicts.dims()[0];
    let time_steps = predicts.dims()[1];
    let features = predicts.dims()[2];

    // Convert to vectors for easier indexing
    let predicts_vec = predicts.clone().into_data().to_vec::<f32>().unwrap();
    let futures_vec = futures.clone().into_data().to_vec::<f32>().unwrap();

    let samples_to_plot = std::cmp::min(10, batch_size);

    for i in 0..samples_to_plot {
        let file_name = format!("{exp_root_path}/test/prediction_{}.png", i);
        let root = BitMapBackend::new(&file_name, (1024, 768)).into_drawing_area();
        root.fill(&WHITE).unwrap();

        let feature_idx: usize = features - 1; // Plot the last feature

        let mut pred_series = Vec::new();
        let mut true_series = Vec::new();

        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;

        for t in 0..time_steps {
            // Flattened index: b * (T * F) + t * F + f
            let offset = i * (time_steps * features) + t * features + feature_idx;
            let p_val = predicts_vec[offset];
            let t_val = futures_vec[offset];

            pred_series.push((t as f32, p_val));
            true_series.push((t as f32, t_val));

            if p_val < min_y {
                min_y = p_val;
            }
            if p_val > max_y {
                max_y = p_val;
            }
            if t_val < min_y {
                min_y = t_val;
            }
            if t_val > max_y {
                max_y = t_val;
            }
        }

        // Add some margin to Y axis
        let y_margin = (max_y - min_y) * 0.1;
        let y_range = (min_y - y_margin)..(max_y + y_margin);

        let mut chart = ChartBuilder::on(&root)
            .caption(
                format!("Sample {} Prediction vs Ground Truth", i),
                ("sans-serif", 20).into_font(),
            )
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(30)
            .build_cartesian_2d(0f32..time_steps as f32, y_range)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        chart
            .draw_series(LineSeries::new(pred_series, &BLUE))
            .unwrap()
            .label("Prediction")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        chart
            .draw_series(LineSeries::new(true_series, &RED))
            .unwrap()
            .label("Ground Truth")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()
            .unwrap();

        root.present().unwrap();
        println!("Saved plot to {}", file_name);
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Distribution;
    use burn_ndarray::NdArray;

    use crate::env_path::get_result_root_path;

    use super::*;

    #[test]
    fn test_save_results() {
        type B = NdArray;

        // Create a persistent directory for output
        let exp_root_path = &get_result_root_path();
        fs::create_dir_all(exp_root_path).unwrap();

        // Create dummy data
        // Batch=20, Time=96, Features=2
        let batch_size = 20;
        let time_steps = 96;
        let features = 2;

        let device = Default::default();

        // Generate random data for predicts and futures
        let predicts: Tensor<B, 3> = Tensor::random(
            [batch_size, time_steps, features],
            Distribution::Uniform(0.0, 10.0),
            &device,
        );
        let futures: Tensor<B, 3> = Tensor::random(
            [batch_size, time_steps, features],
            Distribution::Uniform(0.0, 10.0),
            &device,
        );

        let error = predicts.clone() - futures.clone();

        save_results(exp_root_path, error, predicts, futures);

        // Verify files are created
        let test_dir = std::path::Path::new(exp_root_path).join("test");
        assert!(test_dir.exists());
        assert!(test_dir.join("mse.csv").exists());
        assert!(test_dir.join("mae.csv").exists());
        assert!(test_dir.join("results.txt").exists());

        // Check that 10 prediction plots are created (min(10, 20) = 10)
        for i in 0..10 {
            assert!(test_dir.join(format!("prediction_{}.png", i)).exists());
        }

        // Optional: Verify content of results.txt
        let results_content = std::fs::read_to_string(test_dir.join("results.txt")).unwrap();
        assert!(results_content.contains("MSE:"));
        assert!(results_content.contains("MAE:"));
        assert!(results_content.contains("RMSE:"));
        assert!(results_content.contains("MAPE:"));
        assert!(results_content.contains("MSPE:"));

        println!("Test output saved to: {}", test_dir.display());
    }
}
