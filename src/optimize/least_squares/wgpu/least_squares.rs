//! WebGPU implementation of least squares algorithms.
//!
//! Implements the [`LeastSquaresAlgorithms`] trait for WebGPU runtime.
//! All implementations delegate to the generic implementations in `impl_generic/least_squares/`.

use crate::optimize::least_squares::impl_generic::{least_squares_impl, leastsq_impl};
use crate::optimize::least_squares::traits::{
    LeastSquaresAlgorithms, LeastSquaresOptions, LeastSquaresTensorResult,
};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl LeastSquaresAlgorithms<WgpuRuntime> for WgpuClient {
    fn leastsq<F>(
        &self,
        f: F,
        x0: &Tensor<WgpuRuntime>,
        options: &LeastSquaresOptions,
    ) -> Result<LeastSquaresTensorResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        let result = leastsq_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("wgpu", "leastsq", e.to_string())
        })?;
        Ok(LeastSquaresTensorResult {
            x: result.x,
            residuals: result.residuals,
            cost: result.cost,
            iterations: result.iterations,
            nfev: result.nfev,
            converged: result.converged,
        })
    }

    fn least_squares<F>(
        &self,
        f: F,
        x0: &Tensor<WgpuRuntime>,
        bounds: Option<(&Tensor<WgpuRuntime>, &Tensor<WgpuRuntime>)>,
        options: &LeastSquaresOptions,
    ) -> Result<LeastSquaresTensorResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        let result = least_squares_impl(self, f, x0, bounds, options).map_err(|e| {
            numr::error::Error::backend_limitation("wgpu", "least_squares", e.to_string())
        })?;
        Ok(LeastSquaresTensorResult {
            x: result.x,
            residuals: result.residuals,
            cost: result.cost,
            iterations: result.iterations,
            nfev: result.nfev,
            converged: result.converged,
        })
    }
}
