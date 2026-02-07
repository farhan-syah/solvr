//! CUDA implementation of trust-constr.

use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::optimize::constrained::impl_generic::trust_constr_impl;
use crate::optimize::constrained::traits::{
    Bounds, ConstrainedOptions, ConstrainedResult, Constraint, TrustConstrAlgorithms,
};
use crate::optimize::error::OptimizeResult;

impl TrustConstrAlgorithms<CudaRuntime> for CudaClient {
    fn trust_constr<F>(
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
        trust_constr_impl(self, f, x0, constraints, bounds, options)
    }
}
