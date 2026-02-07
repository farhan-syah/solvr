//! WebGPU implementation of trust-constr.

use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::constrained::impl_generic::trust_constr_impl;
use crate::optimize::constrained::traits::{
    Bounds, ConstrainedOptions, ConstrainedResult, Constraint, TrustConstrAlgorithms,
};
use crate::optimize::error::OptimizeResult;

impl TrustConstrAlgorithms<WgpuRuntime> for WgpuClient {
    fn trust_constr<F>(
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
        trust_constr_impl(self, f, x0, constraints, bounds, options)
    }
}
