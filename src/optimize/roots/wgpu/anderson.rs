//! WebGPU implementation of Anderson mixing.

use crate::optimize::roots::RootTensorResult;
use crate::optimize::roots::impl_generic::anderson_impl;
use crate::optimize::roots::traits::anderson::{AndersonAlgorithms, AndersonOptions};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl AndersonAlgorithms<WgpuRuntime> for WgpuClient {
    fn anderson<G>(
        &self,
        g: G,
        x0: &Tensor<WgpuRuntime>,
        options: &AndersonOptions,
    ) -> Result<RootTensorResult<WgpuRuntime>>
    where
        G: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        let result = anderson_impl(self, g, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("wgpu", "anderson", e.to_string())
        })?;
        Ok(result)
    }
}
