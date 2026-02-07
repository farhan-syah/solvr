//! WebGPU implementation of measurement algorithms.

use crate::morphology::impl_generic::{
    center_of_mass_impl, find_objects_impl, label_impl, mean_labels_impl, sum_labels_impl,
};
use crate::morphology::traits::binary::StructuringElement;
use crate::morphology::traits::measurements::{MeasurementAlgorithms, RegionProperties};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl MeasurementAlgorithms<WgpuRuntime> for WgpuClient {
    fn label(
        &self,
        input: &Tensor<WgpuRuntime>,
        structure: StructuringElement,
    ) -> Result<(Tensor<WgpuRuntime>, usize)> {
        label_impl(self, input, structure)
    }
    fn find_objects(
        &self,
        labels: &Tensor<WgpuRuntime>,
        num_labels: usize,
    ) -> Result<Vec<RegionProperties>> {
        find_objects_impl(self, labels, num_labels)
    }
    fn sum_labels(
        &self,
        input: &Tensor<WgpuRuntime>,
        labels: &Tensor<WgpuRuntime>,
        num_labels: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        sum_labels_impl(self, input, labels, num_labels)
    }
    fn mean_labels(
        &self,
        input: &Tensor<WgpuRuntime>,
        labels: &Tensor<WgpuRuntime>,
        num_labels: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        mean_labels_impl(self, input, labels, num_labels)
    }
    fn center_of_mass(
        &self,
        input: &Tensor<WgpuRuntime>,
        labels: &Tensor<WgpuRuntime>,
        num_labels: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        center_of_mass_impl(self, input, labels, num_labels)
    }
}
