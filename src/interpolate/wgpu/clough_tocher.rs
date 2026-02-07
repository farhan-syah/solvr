use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::clough_tocher::{
    clough_tocher_evaluate_impl, clough_tocher_fit_impl,
};
use crate::interpolate::traits::clough_tocher::{CloughTocher2D, CloughTocher2DAlgorithms};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl CloughTocher2DAlgorithms<WgpuRuntime> for WgpuClient {
    fn clough_tocher_fit(
        &self,
        points: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        fill_value: f64,
    ) -> InterpolateResult<CloughTocher2D<WgpuRuntime>> {
        clough_tocher_fit_impl(self, points, values, fill_value)
    }

    fn clough_tocher_evaluate(
        &self,
        ct: &CloughTocher2D<WgpuRuntime>,
        xi: &Tensor<WgpuRuntime>,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        clough_tocher_evaluate_impl(self, ct, xi)
    }
}
