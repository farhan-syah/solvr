//! CUDA implementation of analog frequency response.

// Allow non-snake_case for `worN` parameter - follows SciPy's naming convention
#![allow(non_snake_case)]

use crate::signal::filter::impl_generic::freqs_impl;
use crate::signal::filter::traits::analog_response::{AnalogResponseAlgorithms, FreqsResult};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl AnalogResponseAlgorithms<CudaRuntime> for CudaClient {
    fn freqs(
        &self,
        b: &Tensor<CudaRuntime>,
        a: &Tensor<CudaRuntime>,
        worN: &Tensor<CudaRuntime>,
        device: &<CudaRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<FreqsResult<CudaRuntime>> {
        freqs_impl(self, b, a, worN, device)
    }
}
