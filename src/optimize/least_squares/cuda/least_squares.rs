//! CUDA implementation of least squares algorithms.
//!
//! Implements the [`LeastSquaresAlgorithms`] trait for CUDA runtime.
//! All implementations delegate to the generic implementations in `impl_generic/least_squares/`.

use crate::optimize::least_squares::impl_generic::{least_squares_impl, leastsq_impl};
use crate::optimize::least_squares::traits::{
    LeastSquaresAlgorithms, LeastSquaresOptions, LeastSquaresTensorResult,
};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl LeastSquaresAlgorithms<CudaRuntime> for CudaClient {
    fn leastsq<F>(
        &self,
        f: F,
        x0: &Tensor<CudaRuntime>,
        options: &LeastSquaresOptions,
    ) -> Result<LeastSquaresTensorResult<CudaRuntime>>
    where
        F: Fn(&Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>>,
    {
        let result = leastsq_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cuda", "leastsq", e.to_string())
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
        x0: &Tensor<CudaRuntime>,
        bounds: Option<(&Tensor<CudaRuntime>, &Tensor<CudaRuntime>)>,
        options: &LeastSquaresOptions,
    ) -> Result<LeastSquaresTensorResult<CudaRuntime>>
    where
        F: Fn(&Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>>,
    {
        let result = least_squares_impl(self, f, x0, bounds, options).map_err(|e| {
            numr::error::Error::backend_limitation("cuda", "least_squares", e.to_string())
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
