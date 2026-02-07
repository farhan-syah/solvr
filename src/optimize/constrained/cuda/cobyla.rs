//! CUDA implementation of COBYLA.

use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::optimize::constrained::impl_generic::cobyla_impl;
use crate::optimize::constrained::traits::{
    Bounds, CobylaAlgorithms, ConstrainedOptions, ConstrainedResult, Constraint,
};
use crate::optimize::error::OptimizeResult;

impl CobylaAlgorithms<CudaRuntime> for CudaClient {
    fn cobyla<F>(
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
        cobyla_impl(self, f, x0, constraints, bounds, options)
    }
}
