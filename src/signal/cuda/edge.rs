//! CUDA implementation of edge detection algorithms.

use crate::signal::impl_generic::{
    gaussian_gradient_magnitude_impl, gaussian_laplace_impl, laplace_impl, prewitt_impl, sobel_impl,
};
use crate::signal::traits::edge::EdgeDetectionAlgorithms;
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl EdgeDetectionAlgorithms<CudaRuntime> for CudaClient {
    fn sobel(&self, input: &Tensor<CudaRuntime>, axis: usize) -> Result<Tensor<CudaRuntime>> {
        sobel_impl(self, input, axis)
    }

    fn prewitt(&self, input: &Tensor<CudaRuntime>, axis: usize) -> Result<Tensor<CudaRuntime>> {
        prewitt_impl(self, input, axis)
    }

    fn laplace(&self, input: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        laplace_impl(self, input)
    }

    fn gaussian_laplace(
        &self,
        input: &Tensor<CudaRuntime>,
        sigma: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        gaussian_laplace_impl(self, input, sigma)
    }

    fn gaussian_gradient_magnitude(
        &self,
        input: &Tensor<CudaRuntime>,
        sigma: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        gaussian_gradient_magnitude_impl(self, input, sigma)
    }
}
