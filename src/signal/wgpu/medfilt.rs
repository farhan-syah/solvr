//! WebGPU implementation of median filter algorithms.
//!
//! Median filtering is CPU-only because computing median requires sorting/selection
//! which doesn't parallelize efficiently on GPU for small sliding windows.

use crate::signal::traits::medfilt::MedianFilterAlgorithms;
use numr::error::{Error, Result};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl MedianFilterAlgorithms<WgpuRuntime> for WgpuClient {
    fn medfilt(
        &self,
        _x: &Tensor<WgpuRuntime>,
        _kernel_size: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        Err(Error::backend_limitation(
            "wgpu",
            "medfilt",
            "Median filtering is CPU-only due to sorting requirements. Transfer data to CPU first.",
        ))
    }

    fn medfilt2d(
        &self,
        _x: &Tensor<WgpuRuntime>,
        _kernel_size: (usize, usize),
    ) -> Result<Tensor<WgpuRuntime>> {
        Err(Error::backend_limitation(
            "wgpu",
            "medfilt2d",
            "Median filtering is CPU-only due to sorting requirements. Transfer data to CPU first.",
        ))
    }
}
