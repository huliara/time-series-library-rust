use burn::{
    module::Module,
    nn::pool::{AvgPool1d, AvgPool1dConfig},
    tensor::{backend::Backend, Tensor},
};
use std::marker::PhantomData;

#[derive(Module, Debug)]
pub struct MovingAvg<B: Backend> {
    kernel_size: usize,
    avg: AvgPool1d,
    phantom: PhantomData<B>,
}

impl<B: Backend> MovingAvg<B> {
    pub fn new(kernel_size: usize, stride: usize) -> Self {
        let avg = AvgPool1dConfig::new(kernel_size)
            .with_stride(stride)
            .with_padding(burn::nn::PaddingConfig1d::Explicit(0))
            .init();
        Self {
            kernel_size,
            avg,
            phantom: PhantomData,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // x: [batch_size, seq_len, channels]
        let x_len = x.dims()[1];

        let front_size = (self.kernel_size - 1) / 2;

        // Burn doesn't have a simple repeat on a specific dim syntax everywhere, but we can use cat?
        // Actually, repeat along a dimension:
        // x.slice([0..batch, 0..1, 0..channels]) -> shape [batch, 1, channels]
        // repeat implies calling repeat on the tensor.
        // Assuming burn Tensor has `repeat` or we construct it.

        let batch_size = x.dims()[0];
        let channels = x.dims()[2];

        let first_val = x.clone().slice([0..batch_size, 0..1, 0..channels]);
        let last_val = x
            .clone()
            .slice([0..batch_size, (x_len - 1)..x_len, 0..channels]);

        let front = first_val.repeat_dim(1, front_size);
        // Note: Python logic:
        // front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)

        // Wait, AvgPool1d logic naturally reduces size if no padding.
        // The padding here is "replication padding" essentially.

        // Python code handles typically odd kernel sizes correctly.
        // If kernel_size=25, front_size=12.

        let x_padded = if front_size > 0 {
            let end = last_val.repeat_dim(1, front_size);
            Tensor::cat(vec![front, x, end], 1)
        } else {
            x
        };

        // x_padded: [batch, padded_len, channels]
        // AvgPool1d expects [batch, channels, length]
        let x_perm = x_padded.swap_dims(1, 2);

        let out = self.avg.forward(x_perm);

        // out: [batch, channels, length]
        // return [batch, length, channels]
        out.swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
pub struct SeriesDecomp<B: Backend> {
    moving_avg: MovingAvg<B>,
}

impl<B: Backend> SeriesDecomp<B> {
    pub fn new(kernel_size: usize) -> Self {
        Self {
            moving_avg: MovingAvg::<B>::new(kernel_size, 1),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let moving_mean = self.moving_avg.forward(x.clone());
        let res = x - moving_mean.clone();
        (res, moving_mean)
    }
}
