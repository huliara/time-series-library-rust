use crate::args::exp::TaskName;
use crate::layers::decomposition::SeriesDecomp;
use crate::models::traits::{AnomalyDetection, Classification, Forecast, Imputation};
use burn::nn::Initializer;
use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};
use clap::Args;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, Args)]
pub struct DLinearArgs {
    pub seq_len: usize,
    pub pred_len: usize,
    pub enc_in: usize,
    pub individual: bool,
    pub moving_avg: usize,
    pub num_class: usize, // Only used for classification task, ignored otherwise
}
#[derive(Config, Debug)]
pub struct DLinearConfig {
    pub args: DLinearArgs,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

impl DLinearConfig {
    pub fn init<B: Backend>(self, task_name: TaskName, device: &B::Device) -> DLinear<B> {
        let config = &self.args;
        let seq_len = config.seq_len;
        let pred_len = match task_name {
            TaskName::LongTermForecast => config.pred_len,
            TaskName::ShortTermForecast => config.pred_len,
            _ => config.seq_len,
        };

        let decomposition: SeriesDecomp<B> = SeriesDecomp::new(config.moving_avg);
        let individual = config.individual;
        let enc_in = config.enc_in;

        let mut shared_seasonal = None;
        let mut shared_trend = None;
        let mut individual_seasonal_weight = None;
        let mut individual_trend_weight = None;
        let mut individual_seasonal_bias = None;
        let mut individual_trend_bias = None;

        if !individual {
            let mut s = LinearConfig::new(seq_len, pred_len)
                .with_initializer(self.initializer.clone())
                .init(device);
            let mut t = LinearConfig::new(seq_len, pred_len)
                .with_initializer(self.initializer.clone())
                .init(device);

            // Init weights: (1/seq_len) * ones
            // Burn Linear weights are [in, out]. Python: [out, in] (printed as such) but stored as [out, in] usually in PyTorch?
            // Wait, PyTorch nn.Linear(in, out) has weight shape [out, in].
            // Burn nn.Linear(in, out) has weight shape [in, out].
            // Python initialization:
            // self.Linear_Seasonal.weight = (1/self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            // So in Burn we need [seq_len, pred_len] filled with 1/seq_len.

            let init_val = 1.0 / (seq_len as f64);
            let w_shape = [seq_len, pred_len];
            let w = Tensor::ones(w_shape, device).mul_scalar(init_val);

            // We need to overwrite the initialized weights using Record or just swapping?
            // Burn Linear fields are private but we can perform set_weight if we had access or we just overwrite via creation?
            // Accessing `.weight` on Linear is `Param<Tensor>`. We can just reassign.
            // Wait, Linear struct definition: pub weight: Param<Tensor<B, 2>>. Yes, it is public.
            s.weight = Param::from_tensor(w.clone());
            t.weight = Param::from_tensor(w);

            // Bias in PyTorch defaults to initialized?
            // Python code doesn't explicitly init bias, so it uses default PyTorch init (uniform).
            // Burn initializes bias to zeros by default? check LinearConfig.
            // LinearConfig default uses Kaiming/Xavier or something?
            // I'll leave bias as default initialized by Burn unless PyTorch code does something specific.
            // The python code only sets .weight.

            shared_seasonal = Some(s);
            shared_trend = Some(t);
        } else {
            // Individual
            // Weights: [channels, output_len, input_len] -> based on python [out, in]
            // But for our matmul logic `x @ w.T`, we need to decide shape.
            // x: [batch, channels, seq_len]
            // We want [batch, channels, pred_len].

            // If we define weight W as [channels, seq_len, pred_len].
            // Then x * W (elementwise) is not right.f

            // Per channel c: y_c = x_c (1 x seq) @ W_c (seq x pred).
            // W shape: [channels, seq_len, pred_len].
            // x shape: [batch, channels, seq_len] -> [batch, channels, 1, seq_len].
            // W shape broadcast: [1, channels, seq_len, pred_len].
            // Matmul: [batch, channels, 1, pred_len].

            // Correct. So we store W as [channels, seq_len, pred_len].
            // Initialization: 1/seq_len

            let init_val = 1.0 / (seq_len as f64);
            let shape = [enc_in, seq_len, pred_len];
            let w_s = Tensor::ones(shape, device).mul_scalar(init_val);
            let w_t = Tensor::ones(shape, device).mul_scalar(init_val);

            individual_seasonal_weight = Some(Param::from_tensor(w_s));
            individual_trend_weight = Some(Param::from_tensor(w_t));

            // Bias: [channels, pred_len]
            let b_shape = [enc_in, pred_len];
            let b_s = Tensor::zeros(b_shape, device); // Zero init for bias is standard-ish or random. Python uses default.
            let b_t = Tensor::zeros(b_shape, device);

            individual_seasonal_bias = Some(Param::from_tensor(b_s));
            individual_trend_bias = Some(Param::from_tensor(b_t));
        }

        let projection = if task_name == TaskName::Classification {
            Some(
                LinearConfig::new(enc_in * seq_len, config.num_class)
                    .with_initializer(self.initializer.clone())
                    .init(device),
            )
        } else {
            None
        };

        DLinear {
            decomposition,
            shared_seasonal,
            shared_trend,
            individual_seasonal_weight,
            individual_trend_weight,
            individual_seasonal_bias,
            individual_trend_bias,
            projection,
            individual,
            enc_in,
            seq_len,
            pred_len,
        }
    }
}

#[derive(Module, Debug)]
pub struct DLinear<B: Backend> {
    decomposition: SeriesDecomp<B>,

    shared_seasonal: Option<Linear<B>>,
    shared_trend: Option<Linear<B>>,

    individual_seasonal_weight: Option<Param<Tensor<B, 3>>>,
    individual_trend_weight: Option<Param<Tensor<B, 3>>>,
    individual_seasonal_bias: Option<Param<Tensor<B, 2>>>,
    individual_trend_bias: Option<Param<Tensor<B, 2>>>,

    projection: Option<Linear<B>>,
    individual: bool,
    enc_in: usize,
    seq_len: usize,
    pred_len: usize,
}

impl<B: Backend> DLinear<B> {
    fn encoder(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // x: [batch, seq_len, channels]
        let (seasonal_init, trend_init) = self.decomposition.forward(x);

        // seasonal_init: [batch, seq_len, channels]
        // Python: seasonal_init.permute(0,2,1) -> [batch, channels, seq_len]

        let seasonal_init = seasonal_init.swap_dims(1, 2);
        let trend_init = trend_init.swap_dims(1, 2);

        let (seasonal_output, trend_output) = if self.individual {
            let ws = self.individual_seasonal_weight.as_ref().unwrap().val();
            let wt = self.individual_trend_weight.as_ref().unwrap().val();
            let bs = self.individual_seasonal_bias.as_ref().unwrap().val();
            let bt = self.individual_trend_bias.as_ref().unwrap().val();

            // init: [batch, channels, seq_len]
            // ws: [channels, seq_len, pred_len]

            // x: [batch, channels, 1, seq_len]
            let s_in = seasonal_init.clone().unsqueeze_dim::<4>(2);
            let t_in = trend_init.clone().unsqueeze_dim::<4>(2);

            // ws: [1, channels, seq_len, pred_len]
            let ws_exp = ws.unsqueeze_dim::<4>(0);
            let wt_exp = wt.unsqueeze_dim::<4>(0);

            // matmul -> [batch, channels, 1, pred_len]
            // flatten -> [batch, channels, pred_len]
            let s_out = s_in.matmul(ws_exp).flatten::<3>(2, 3);
            let t_out = t_in.matmul(wt_exp).flatten::<3>(2, 3);

            // Add bias [channels, pred_len] -> broadcast to [batch, channels, pred_len]
            let bs_exp = bs.unsqueeze_dim::<3>(0);
            let bt_exp = bt.unsqueeze_dim::<3>(0);

            (s_out + bs_exp, t_out + bt_exp)
        } else {
            let s_layer = self.shared_seasonal.as_ref().unwrap();
            let t_layer = self.shared_trend.as_ref().unwrap();

            // layers expect [..., in_dim].
            // inputs are [batch, channels, seq_len].
            // output [batch, channels, pred_len].
            (s_layer.forward(seasonal_init), t_layer.forward(trend_init))
        };

        let x = seasonal_output + trend_output;
        // Python: return x.permute(0,2,1) -> [batch, pred_len, channels]
        x.swap_dims(1, 2)
    }
}

impl<B: Backend> Forecast<B> for DLinear<B> {
    fn forecast(
        &self,
        x: Tensor<B, 3>,
        _x_mark: Tensor<B, 3>,
        _y: Tensor<B, 3>,
        _y_mark: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.encoder(x)
    }
}

impl<B: Backend> Imputation<B> for DLinear<B> {
    fn imputation(
        &self,
        x: Tensor<B, 3>,
        _x_mark: Tensor<B, 3>,
        _y: Tensor<B, 3>,
        _y_mark: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.encoder(x)
    }
}

impl<B: Backend> AnomalyDetection<B> for DLinear<B> {
    fn anomaly_detection(
        &self,
        x: Tensor<B, 3>,
        _x_mark: Tensor<B, 3>,
        _y: Tensor<B, 3>,
        _y_mark: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        self.encoder(x)
    }
}

impl<B: Backend> Classification<B> for DLinear<B> {
    fn classification(
        &self,
        x: Tensor<B, 3>,
        _x_mark: Tensor<B, 3>,
        _y: Tensor<B, 3>,
        _y_mark: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let enc_out = self.encoder(x);
        // enc_out: [batch, seq_len, channels]
        // Flatten -> [batch, seq_len * channels]

        let batch_size = enc_out.dims()[0];
        let flattened = enc_out.reshape([batch_size, self.seq_len * self.enc_in]);

        self.projection
            .as_ref()
            .unwrap()
            .forward(flattened)
            .unsqueeze_dim::<3>(1)
    }
}

#[cfg(test)]
mod tests {
    use super::{DLinear, DLinearConfig};
    use crate::args::exp::TaskName;
    use crate::models::dlinear::DLinearArgs;
    use crate::models::test_util::assert_module_forecast;
    use burn::backend::wgpu::Wgpu;
    use burn::nn::Initializer;
    #[test]
    fn test_dlinear_forecast() {
        type B = Wgpu;
        let device = Default::default();
        let task_name = TaskName::LongTermForecast;
        let args = DLinearArgs {
            seq_len: 96,
            pred_len: 96,
            enc_in: 7,
            individual: false,
            moving_avg: 25,
            num_class: 10,
        };
        let initializer = Initializer::Constant { value: (0.01) };
        let model = DLinearConfig::new(args)
            .with_initializer(initializer)
            .init(task_name, &device);

        assert_module_forecast::<B, DLinear<B>>(model);
    }
}
