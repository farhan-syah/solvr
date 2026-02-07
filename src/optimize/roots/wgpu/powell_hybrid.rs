//! WebGPU implementation of Powell hybrid method.

use crate::optimize::roots::impl_generic::powell_hybrid_impl;
use crate::optimize::roots::traits::PowellHybridAlgorithms;
use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl PowellHybridAlgorithms<WgpuRuntime> for WgpuClient {
    fn powell_hybrid<F>(
        &self,
        f: F,
        x0: &Tensor<WgpuRuntime>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<WgpuRuntime>>
    where
        F: Fn(&Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>>,
    {
        let result = powell_hybrid_impl(self, f, x0, options).map_err(|e| {
            numr::error::Error::backend_limitation("wgpu", "powell_hybrid", e.to_string())
        })?;
        Ok(result)
    }
}
