//! WebGPU implementation of simulated annealing.

use numr::error::Result;
use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;

use crate::optimize::global::GlobalOptions;
use crate::optimize::global::impl_generic::simulated_annealing::simulated_annealing_impl;
use crate::optimize::global::traits::SimulatedAnnealingAlgorithms;
use crate::optimize::global::traits::simulated_annealing::SimulatedAnnealingResult;
use numr::runtime::wgpu::WgpuClient;

impl SimulatedAnnealingAlgorithms<WgpuRuntime> for WgpuClient {
    fn simulated_annealing<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<WgpuRuntime>,
        upper_bounds: &Tensor<WgpuRuntime>,
        options: &GlobalOptions,
    ) -> Result<SimulatedAnnealingResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<f64>,
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
