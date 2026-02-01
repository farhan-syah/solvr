//! CUDA implementation of differential evolution.

use numr::error::Result;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

use crate::optimize::global::GlobalOptions;
use crate::optimize::global::impl_generic::differential_evolution::differential_evolution_impl;
use crate::optimize::global::traits::DifferentialEvolutionAlgorithms;
use crate::optimize::global::traits::differential_evolution::DifferentialEvolutionResult;
use numr::runtime::cuda::CudaClient;

impl DifferentialEvolutionAlgorithms<CudaRuntime> for CudaClient {
    fn differential_evolution<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<CudaRuntime>,
        upper_bounds: &Tensor<CudaRuntime>,
        options: &GlobalOptions,
    ) -> Result<DifferentialEvolutionResult<CudaRuntime>>
    where
        F: Fn(&Tensor<CudaRuntime>) -> Result<f64>,
    {
        let result = differential_evolution_impl(self, f, lower_bounds, upper_bounds, options)?;
        Ok(DifferentialEvolutionResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            nfev: result.nfev,
            converged: result.converged,
        })
    }
}
