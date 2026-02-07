//! CUDA implementation of Anderson mixing.

use crate::optimize::roots::RootTensorResult;
use crate::optimize::roots::impl_generic::anderson_impl;
use crate::optimize::roots::traits::anderson::{AndersonAlgorithms, AndersonOptions};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl AndersonAlgorithms<CudaRuntime> for CudaClient {
    fn anderson<G>(
        &self,
        g: G,
        x0: &Tensor<CudaRuntime>,
        options: &AndersonOptions,
    ) -> Result<RootTensorResult<CudaRuntime>>
    where
        G: Fn(&Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>>,
    {
        let result = anderson_impl(self, g, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("cuda", "anderson", e.to_string())
        })?;
        Ok(result)
    }
}
