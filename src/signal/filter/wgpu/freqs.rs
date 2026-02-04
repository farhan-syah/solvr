//! WebGPU implementation of analog frequency response.

// Allow non-snake_case for `worN` parameter - follows SciPy's naming convention
#![allow(non_snake_case)]

use crate::signal::filter::impl_generic::freqs_impl;
use crate::signal::filter::traits::analog_response::{AnalogResponseAlgorithms, FreqsResult};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl AnalogResponseAlgorithms<WgpuRuntime> for WgpuClient {
    fn freqs(
        &self,
        b: &Tensor<WgpuRuntime>,
        a: &Tensor<WgpuRuntime>,
        worN: &Tensor<WgpuRuntime>,
        device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<FreqsResult<WgpuRuntime>> {
        freqs_impl(self, b, a, worN, device)
    }
}
