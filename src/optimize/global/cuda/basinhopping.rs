//! CUDA implementation of basin-hopping.

use numr::error::Result;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

use crate::optimize::global::GlobalOptions;
use crate::optimize::global::impl_generic::basinhopping::basinhopping_impl;
use crate::optimize::global::traits::BasinHoppingAlgorithms;
use crate::optimize::global::traits::basin_hopping::BasinHoppingResult;
use numr::runtime::cuda::CudaClient;

impl BasinHoppingAlgorithms<CudaRuntime> for CudaClient {
    fn basinhopping<F>(
        &self,
        f: F,
        x0: &Tensor<CudaRuntime>,
        lower_bounds: &Tensor<CudaRuntime>,
        upper_bounds: &Tensor<CudaRuntime>,
        options: &GlobalOptions,
    ) -> Result<BasinHoppingResult<CudaRuntime>>
    where
        F: Fn(&Tensor<CudaRuntime>) -> Result<f64>,
    {
        let result =
            basinhopping_impl(self, f, x0, lower_bounds, upper_bounds, options).map_err(|e| {
                numr::error::Error::backend_limitation("cuda", "basinhopping", e.to_string())
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
