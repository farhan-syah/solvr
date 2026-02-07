//! CUDA implementation of distance transform algorithms.

use crate::spatial::impl_generic::distance_transform::{
    distance_transform_edt_impl, distance_transform_impl,
};
use crate::spatial::traits::distance_transform::{
    DistanceTransformAlgorithms, DistanceTransformMetric,
};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl DistanceTransformAlgorithms<CudaRuntime> for CudaClient {
    fn distance_transform(
        &self,
        input: &Tensor<CudaRuntime>,
        metric: DistanceTransformMetric,
    ) -> Result<Tensor<CudaRuntime>> {
        distance_transform_impl(self, input, metric)
    }

    fn distance_transform_edt(&self, input: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        distance_transform_edt_impl(self, input)
    }
}
