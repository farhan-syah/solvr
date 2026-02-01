//! CUDA implementation of Newton's method for root finding.

use crate::optimize::roots::impl_generic::newton_system_impl;
use crate::optimize::roots::traits::NewtonSystemAlgorithms;
use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl NewtonSystemAlgorithms<CudaRuntime> for CudaClient {
    fn newton_system<F>(
        &self,
        f: F,
        x0: &Tensor<CudaRuntime>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<CudaRuntime>>
    where
        F: Fn(&Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>>,
    {
        let result = newton_system_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cuda", "newton_system", e.to_string())
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
