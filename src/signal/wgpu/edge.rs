//! WebGPU implementation of edge detection algorithms.

use crate::signal::impl_generic::{
    gaussian_gradient_magnitude_impl, gaussian_laplace_impl, laplace_impl, prewitt_impl, sobel_impl,
};
use crate::signal::traits::edge::EdgeDetectionAlgorithms;
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl EdgeDetectionAlgorithms<WgpuRuntime> for WgpuClient {
    fn sobel(&self, input: &Tensor<WgpuRuntime>, axis: usize) -> Result<Tensor<WgpuRuntime>> {
        sobel_impl(self, input, axis)
    }

    fn prewitt(&self, input: &Tensor<WgpuRuntime>, axis: usize) -> Result<Tensor<WgpuRuntime>> {
        prewitt_impl(self, input, axis)
    }

    fn laplace(&self, input: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        laplace_impl(self, input)
    }

    fn gaussian_laplace(
        &self,
        input: &Tensor<WgpuRuntime>,
        sigma: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        gaussian_laplace_impl(self, input, sigma)
    }

    fn gaussian_gradient_magnitude(
        &self,
        input: &Tensor<WgpuRuntime>,
        sigma: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        gaussian_gradient_magnitude_impl(self, input, sigma)
    }
}
