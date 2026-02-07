//! CUDA implementation of measurement algorithms.

use crate::morphology::impl_generic::{
    center_of_mass_impl, find_objects_impl, label_impl, mean_labels_impl, sum_labels_impl,
};
use crate::morphology::traits::binary::StructuringElement;
use crate::morphology::traits::measurements::{MeasurementAlgorithms, RegionProperties};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl MeasurementAlgorithms<CudaRuntime> for CudaClient {
    fn label(
        &self,
        input: &Tensor<CudaRuntime>,
        structure: StructuringElement,
    ) -> Result<(Tensor<CudaRuntime>, usize)> {
        label_impl(self, input, structure)
    }
    fn find_objects(
        &self,
        labels: &Tensor<CudaRuntime>,
        num_labels: usize,
    ) -> Result<Vec<RegionProperties>> {
        find_objects_impl(self, labels, num_labels)
    }
    fn sum_labels(
        &self,
        input: &Tensor<CudaRuntime>,
        labels: &Tensor<CudaRuntime>,
        num_labels: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        sum_labels_impl(self, input, labels, num_labels)
    }
    fn mean_labels(
        &self,
        input: &Tensor<CudaRuntime>,
        labels: &Tensor<CudaRuntime>,
        num_labels: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        mean_labels_impl(self, input, labels, num_labels)
    }
    fn center_of_mass(
        &self,
        input: &Tensor<CudaRuntime>,
        labels: &Tensor<CudaRuntime>,
        num_labels: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        center_of_mass_impl(self, input, labels, num_labels)
    }
}
