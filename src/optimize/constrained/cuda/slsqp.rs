//! CUDA implementation of SLSQP.

use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::optimize::constrained::impl_generic::slsqp_impl;
use crate::optimize::constrained::traits::{
    Bounds, ConstrainedOptions, ConstrainedResult, Constraint, SlsqpAlgorithms,
};
use crate::optimize::error::OptimizeResult;

impl SlsqpAlgorithms<CudaRuntime> for CudaClient {
    fn slsqp<F>(
        &self,
        f: F,
        x0: &Tensor<CudaRuntime>,
        constraints: &[Constraint<'_, CudaRuntime>],
        bounds: &Bounds<CudaRuntime>,
        options: &ConstrainedOptions,
    ) -> OptimizeResult<ConstrainedResult<CudaRuntime>>
    where
        F: Fn(&Tensor<CudaRuntime>) -> Result<f64>,
    {
        slsqp_impl(self, f, x0, constraints, bounds, options)
    }
}
