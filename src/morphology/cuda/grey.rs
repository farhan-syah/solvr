//! CUDA implementation of grey-scale morphology algorithms.

use crate::morphology::impl_generic::{
    black_tophat_impl, grey_closing_impl, grey_dilation_impl, grey_erosion_impl, grey_opening_impl,
    morphological_gradient_impl, white_tophat_impl,
};
use crate::morphology::traits::grey::GreyMorphologyAlgorithms;
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl GreyMorphologyAlgorithms<CudaRuntime> for CudaClient {
    fn grey_erosion(
        &self,
        input: &Tensor<CudaRuntime>,
        size: &[usize],
    ) -> Result<Tensor<CudaRuntime>> {
        grey_erosion_impl(self, input, size)
    }
    fn grey_dilation(
        &self,
        input: &Tensor<CudaRuntime>,
        size: &[usize],
    ) -> Result<Tensor<CudaRuntime>> {
        grey_dilation_impl(self, input, size)
    }
    fn grey_opening(
        &self,
        input: &Tensor<CudaRuntime>,
        size: &[usize],
    ) -> Result<Tensor<CudaRuntime>> {
        grey_opening_impl(self, input, size)
    }
    fn grey_closing(
        &self,
        input: &Tensor<CudaRuntime>,
        size: &[usize],
    ) -> Result<Tensor<CudaRuntime>> {
        grey_closing_impl(self, input, size)
    }
    fn morphological_gradient(
        &self,
        input: &Tensor<CudaRuntime>,
        size: &[usize],
    ) -> Result<Tensor<CudaRuntime>> {
        morphological_gradient_impl(self, input, size)
    }
    fn white_tophat(
        &self,
        input: &Tensor<CudaRuntime>,
        size: &[usize],
    ) -> Result<Tensor<CudaRuntime>> {
        white_tophat_impl(self, input, size)
    }
    fn black_tophat(
        &self,
        input: &Tensor<CudaRuntime>,
        size: &[usize],
    ) -> Result<Tensor<CudaRuntime>> {
        black_tophat_impl(self, input, size)
    }
}
