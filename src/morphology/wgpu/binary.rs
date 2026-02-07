//! WebGPU implementation of binary morphology algorithms.

use crate::morphology::impl_generic::{
    binary_closing_impl, binary_dilation_impl, binary_erosion_impl, binary_fill_holes_impl,
    binary_opening_impl,
};
use crate::morphology::traits::binary::{BinaryMorphologyAlgorithms, StructuringElement};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl BinaryMorphologyAlgorithms<WgpuRuntime> for WgpuClient {
    fn binary_erosion(
        &self,
        input: &Tensor<WgpuRuntime>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        binary_erosion_impl(self, input, structure, iterations)
    }

    fn binary_dilation(
        &self,
        input: &Tensor<WgpuRuntime>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        binary_dilation_impl(self, input, structure, iterations)
    }

    fn binary_opening(
        &self,
        input: &Tensor<WgpuRuntime>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        binary_opening_impl(self, input, structure, iterations)
    }

    fn binary_closing(
        &self,
        input: &Tensor<WgpuRuntime>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        binary_closing_impl(self, input, structure, iterations)
    }

    fn binary_fill_holes(&self, input: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        binary_fill_holes_impl(self, input)
    }
}
