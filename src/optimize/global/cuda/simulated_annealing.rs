//! CUDA implementation of simulated annealing.

use numr::error::Result;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

use crate::optimize::global::GlobalOptions;
use crate::optimize::global::impl_generic::simulated_annealing::simulated_annealing_impl;
use crate::optimize::global::traits::SimulatedAnnealingAlgorithms;
use crate::optimize::global::traits::simulated_annealing::SimulatedAnnealingResult;
use numr::runtime::cuda::CudaClient;

impl SimulatedAnnealingAlgorithms<CudaRuntime> for CudaClient {
    fn simulated_annealing<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<CudaRuntime>,
        upper_bounds: &Tensor<CudaRuntime>,
        options: &GlobalOptions,
    ) -> Result<SimulatedAnnealingResult<CudaRuntime>>
    where
        F: Fn(&Tensor<CudaRuntime>) -> Result<f64>,
    {
        let result = simulated_annealing_impl(self, f, lower_bounds, upper_bounds, options)?;
        Ok(SimulatedAnnealingResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            nfev: result.nfev,
            converged: result.converged,
        })
    }
}
