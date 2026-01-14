use burn::module::Module;
use burn::tensor::{backend::Backend, Tensor};
/// Pads the input tensor using replication of the boundary elements.
///
/// Corresponds to `torch.nn.ReplicationPad1d`.
/// Expected input shape: [batch_size, channels, length]
#[derive(Module, Debug, Clone)]
pub struct ReplicationPad1d {
    pub padding: (usize, usize),
}

impl ReplicationPad1d {
    /// Creates a new ReplicationPad1d module.
    ///
    /// # Arguments
    ///
    /// * `padding` - A tuple (left, right) specifying the amount of padding.
    pub fn new(padding: (usize, usize)) -> Self {
        Self { padding }
    }

    /// Applies replication padding to the input tensor.
    ///
    /// # Arguments
    ///
    /// * `x` - Input tensor of shape [..., length]
    ///
    /// # Returns
    ///
    /// Output tensor with the last dimension padded.
    pub fn forward<B: Backend, const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let (pad_left, pad_right) = self.padding;

        if pad_left == 0 && pad_right == 0 {
            return x;
        }

        let dims: [usize; D] = x.dims();
        let last_dim_idx = D - 1;
        let length = dims[last_dim_idx];
        let mut parts = Vec::with_capacity(3);

        // Left padding
        if pad_left > 0 {
            // Take the first element along the last dimension
            let ranges: [std::ops::Range<usize>; D] =
                core::array::from_fn(|i| if i == last_dim_idx { 0..1 } else { 0..dims[i] });
            let first = x.clone().slice(ranges);
            // Repeat it `pad_left` times along the last dimension
            let left_pad = first.repeat_dim(last_dim_idx, pad_left);
            parts.push(left_pad);
        }

        // Original tensor
        parts.push(x.clone());

        // Right padding
        if pad_right > 0 {
            // Take the last element along the last dimension
            let ranges: [std::ops::Range<usize>; D] = core::array::from_fn(|i| {
                if i == last_dim_idx {
                    length - 1..length
                } else {
                    0..dims[i]
                }
            });
            let last = x.clone().slice(ranges);
            // Repeat it `pad_right` times along the last dimension
            let right_pad = last.repeat_dim(last_dim_idx, pad_right);
            parts.push(right_pad);
        }

        Tensor::cat(parts, last_dim_idx)
    }
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tensor;
    use burn_wgpu::Wgpu;

    use crate::layers::replication_pad_1d::ReplicationPad1d;

    #[test]
    fn test_replication_pad_1d() {
        let tensor = Tensor::<Wgpu, 4>::from_data(
            [
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                ],
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                ],
            ],
            &Default::default(),
        );
        let answer = Tensor::<Wgpu, 4>::from_data(
            [
                [
                    [
                        [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
                        [4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                    ],
                    [
                        [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
                        [4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                    ],
                ],
                [
                    [
                        [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
                        [4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                    ],
                    [
                        [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
                        [4.0, 4.0, 4.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                    ],
                ],
            ],
            &Default::default(),
        );
        let pad = ReplicationPad1d::new((2, 3));

        let output = pad.forward(tensor.clone());
        assert_eq!(output.to_data(), answer.to_data());
    }
}
