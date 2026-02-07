//! WebGPU implementation of SLSQP.

use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::constrained::impl_generic::slsqp_impl;
use crate::optimize::constrained::traits::{
    Bounds, ConstrainedOptions, ConstrainedResult, Constraint, SlsqpAlgorithms,
};
use crate::optimize::error::OptimizeResult;

impl SlsqpAlgorithms<WgpuRuntime> for WgpuClient {
    fn slsqp<F>(
        &self,
        f: F,
        x0: &Tensor<WgpuRuntime>,
        constraints: &[Constraint<'_, WgpuRuntime>],
        bounds: &Bounds<WgpuRuntime>,
        options: &ConstrainedOptions,
    ) -> OptimizeResult<ConstrainedResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<f64>,
    {
        slsqp_impl(self, f, x0, constraints, bounds, options)
    }
}
