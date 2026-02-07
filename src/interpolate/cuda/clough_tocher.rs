use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::clough_tocher::{
    clough_tocher_evaluate_impl, clough_tocher_fit_impl,
};
use crate::interpolate::traits::clough_tocher::{CloughTocher2D, CloughTocher2DAlgorithms};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl CloughTocher2DAlgorithms<CudaRuntime> for CudaClient {
    fn clough_tocher_fit(
        &self,
        points: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        fill_value: f64,
    ) -> InterpolateResult<CloughTocher2D<CudaRuntime>> {
        clough_tocher_fit_impl(self, points, values, fill_value)
    }

    fn clough_tocher_evaluate(
        &self,
        ct: &CloughTocher2D<CudaRuntime>,
        xi: &Tensor<CudaRuntime>,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        clough_tocher_evaluate_impl(self, ct, xi)
    }
}
