//! WebGPU implementation of basin-hopping.

use numr::error::Result;
use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;

use crate::optimize::global::GlobalOptions;
use crate::optimize::global::impl_generic::basinhopping::basinhopping_impl;
use crate::optimize::global::traits::BasinHoppingAlgorithms;
use crate::optimize::global::traits::basin_hopping::BasinHoppingResult;
use numr::runtime::wgpu::WgpuClient;

impl BasinHoppingAlgorithms<WgpuRuntime> for WgpuClient {
    fn basinhopping<F>(
        &self,
        f: F,
        x0: &Tensor<WgpuRuntime>,
        lower_bounds: &Tensor<WgpuRuntime>,
        upper_bounds: &Tensor<WgpuRuntime>,
        options: &GlobalOptions,
    ) -> Result<BasinHoppingResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<f64>,
    {
        let result =
            basinhopping_impl(self, f, x0, lower_bounds, upper_bounds, options).map_err(|e| {
                numr::error::Error::backend_limitation("wgpu", "basinhopping", e.to_string())
            })?;
        Ok(BasinHoppingResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            nfev: result.nfev,
            converged: result.converged,
        })
    }
}
