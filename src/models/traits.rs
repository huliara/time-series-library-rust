use burn::{tensor::backend::Backend, Tensor};
pub trait Forecast<B: Backend> {
    fn forecast(
        &self,
        x: Tensor<B, 3>,
        x_mark: Tensor<B, 3>,
        y: Tensor<B, 3>,
        y_mark: Tensor<B, 3>,
    ) -> Tensor<B, 3>;
}
pub trait Imputation<B: Backend> {
    fn imputation(
        &self,
        x: Tensor<B, 3>,
        x_mark: Tensor<B, 3>,
        y: Tensor<B, 3>,
        y_mark: Tensor<B, 3>,
    ) -> Tensor<B, 3>;
}
pub trait AnomalyDetection<B: Backend> {
    fn anomaly_detection(
        &self,
        x: Tensor<B, 3>,
        x_mark: Tensor<B, 3>,
        y: Tensor<B, 3>,
        y_mark: Tensor<B, 3>,
    ) -> Tensor<B, 3>;
}
pub trait Classification<B: Backend> {
    fn classification(
        &self,
        x: Tensor<B, 3>,
        x_mark: Tensor<B, 3>,
        y: Tensor<B, 3>,
        y_mark: Tensor<B, 3>,
    ) -> Tensor<B, 3>;
}
