//! WebGPU implementation of COBYLA.

use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::constrained::impl_generic::cobyla_impl;
use crate::optimize::constrained::traits::{
    Bounds, CobylaAlgorithms, ConstrainedOptions, ConstrainedResult, Constraint,
};
use crate::optimize::error::OptimizeResult;

impl CobylaAlgorithms<WgpuRuntime> for WgpuClient {
    fn cobyla<F>(
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
        cobyla_impl(self, f, x0, constraints, bounds, options)
    }
}
