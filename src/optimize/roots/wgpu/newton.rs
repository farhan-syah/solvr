//! WebGPU implementation of Newton's method for root finding.

use crate::optimize::roots::impl_generic::newton_system_impl;
use crate::optimize::roots::traits::NewtonSystemAlgorithms;
use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl NewtonSystemAlgorithms<WgpuRuntime> for WgpuClient {
    fn newton_system<F>(
        &self,
        f: F,
        x0: &Tensor<WgpuRuntime>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        let result = newton_system_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("wgpu", "newton_system", e.to_string())
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
