//! CUDA implementation of dual annealing.

use numr::error::Result;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

use crate::optimize::global::GlobalOptions;
use crate::optimize::global::impl_generic::dual_annealing::dual_annealing_impl;
use crate::optimize::global::traits::DualAnnealingAlgorithms;
use crate::optimize::global::traits::dual_annealing::DualAnnealingResult;
use numr::runtime::cuda::CudaClient;

impl DualAnnealingAlgorithms<CudaRuntime> for CudaClient {
    fn dual_annealing<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<CudaRuntime>,
        upper_bounds: &Tensor<CudaRuntime>,
        options: &GlobalOptions,
    ) -> Result<DualAnnealingResult<CudaRuntime>>
    where
        F: Fn(&Tensor<CudaRuntime>) -> Result<f64>,
    {
        let result =
            dual_annealing_impl(self, f, lower_bounds, upper_bounds, options).map_err(|e| {
                numr::error::Error::backend_limitation("cuda", "dual_annealing", e.to_string())
            })?;
        Ok(DualAnnealingResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            nfev: result.nfev,
            converged: result.converged,
        })
    }
}
