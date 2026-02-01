//! WebGPU implementation of Levenberg-Marquardt algorithm for root finding.

use crate::optimize::roots::impl_generic::levenberg_marquardt_impl;
use crate::optimize::roots::traits::LevenbergMarquardtAlgorithms;
use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl LevenbergMarquardtAlgorithms<WgpuRuntime> for WgpuClient {
    fn levenberg_marquardt<F>(
        &self,
        f: F,
        x0: &Tensor<WgpuRuntime>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        let result = levenberg_marquardt_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("wgpu", "levenberg_marquardt", e.to_string())
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
