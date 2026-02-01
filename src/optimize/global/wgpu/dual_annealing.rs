//! WebGPU implementation of dual annealing.

use numr::error::Result;
use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;

use crate::optimize::global::GlobalOptions;
use crate::optimize::global::impl_generic::dual_annealing::dual_annealing_impl;
use crate::optimize::global::traits::DualAnnealingAlgorithms;
use crate::optimize::global::traits::dual_annealing::DualAnnealingResult;
use numr::runtime::wgpu::WgpuClient;

impl DualAnnealingAlgorithms<WgpuRuntime> for WgpuClient {
    fn dual_annealing<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<WgpuRuntime>,
        upper_bounds: &Tensor<WgpuRuntime>,
        options: &GlobalOptions,
    ) -> Result<DualAnnealingResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<f64>,
    {
        let result = dual_annealing_impl(self, f, lower_bounds, upper_bounds, options)?;
        Ok(DualAnnealingResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            nfev: result.nfev,
            converged: result.converged,
        })
    }
}
