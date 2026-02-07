//! CUDA implementation of binary morphology algorithms.

use crate::morphology::impl_generic::{
    binary_closing_impl, binary_dilation_impl, binary_erosion_impl, binary_fill_holes_impl,
    binary_opening_impl,
};
use crate::morphology::traits::binary::{BinaryMorphologyAlgorithms, StructuringElement};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl BinaryMorphologyAlgorithms<CudaRuntime> for CudaClient {
    fn binary_erosion(
        &self,
        input: &Tensor<CudaRuntime>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        binary_erosion_impl(self, input, structure, iterations)
    }

    fn binary_dilation(
        &self,
        input: &Tensor<CudaRuntime>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        binary_dilation_impl(self, input, structure, iterations)
    }

    fn binary_opening(
        &self,
        input: &Tensor<CudaRuntime>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        binary_opening_impl(self, input, structure, iterations)
    }

    fn binary_closing(
        &self,
        input: &Tensor<CudaRuntime>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        binary_closing_impl(self, input, structure, iterations)
    }

    fn binary_fill_holes(&self, input: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        binary_fill_holes_impl(self, input)
    }
}
