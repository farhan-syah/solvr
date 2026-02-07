use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::clough_tocher::{
    clough_tocher_evaluate_impl, clough_tocher_fit_impl,
};
use crate::interpolate::traits::clough_tocher::{CloughTocher2D, CloughTocher2DAlgorithms};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl CloughTocher2DAlgorithms<CpuRuntime> for CpuClient {
    fn clough_tocher_fit(
        &self,
        points: &Tensor<CpuRuntime>,
        values: &Tensor<CpuRuntime>,
        fill_value: f64,
    ) -> InterpolateResult<CloughTocher2D<CpuRuntime>> {
        clough_tocher_fit_impl(self, points, values, fill_value)
    }

    fn clough_tocher_evaluate(
        &self,
        ct: &CloughTocher2D<CpuRuntime>,
        xi: &Tensor<CpuRuntime>,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        clough_tocher_evaluate_impl(self, ct, xi)
    }
}
