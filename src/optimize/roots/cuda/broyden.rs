//! CUDA implementation of Broyden's method for root finding.

use crate::optimize::roots::impl_generic::broyden1_impl;
use crate::optimize::roots::traits::Broyden1Algorithms;
use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl Broyden1Algorithms<CudaRuntime> for CudaClient {
    fn broyden1<F>(
        &self,
        f: F,
        x0: &Tensor<CudaRuntime>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<CudaRuntime>>
    where
        F: Fn(&Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>>,
    {
        let result = broyden1_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cuda", "broyden1", e.to_string())
        })?;
        Ok(RootTensorResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            residual_norm: result.residual_norm,
            converged: result.converged,
        })
    }
}
