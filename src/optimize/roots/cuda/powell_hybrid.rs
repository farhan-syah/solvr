//! CUDA implementation of Powell hybrid method.

use crate::optimize::roots::impl_generic::powell_hybrid_impl;
use crate::optimize::roots::traits::PowellHybridAlgorithms;
use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl PowellHybridAlgorithms<CudaRuntime> for CudaClient {
    fn powell_hybrid<F>(
        &self,
        f: F,
        x0: &Tensor<CudaRuntime>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<CudaRuntime>>
    where
        F: Fn(&Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>>,
    {
        let result = powell_hybrid_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cuda", "powell_hybrid", e.to_string())
        })?;
        Ok(result)
    }
}
