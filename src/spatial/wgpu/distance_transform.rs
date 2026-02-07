//! WebGPU implementation of distance transform algorithms.

use crate::spatial::impl_generic::distance_transform::{
    distance_transform_edt_impl, distance_transform_impl,
};
use crate::spatial::traits::distance_transform::{
    DistanceTransformAlgorithms, DistanceTransformMetric,
};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl DistanceTransformAlgorithms<WgpuRuntime> for WgpuClient {
    fn distance_transform(
        &self,
        input: &Tensor<WgpuRuntime>,
        metric: DistanceTransformMetric,
    ) -> Result<Tensor<WgpuRuntime>> {
        distance_transform_impl(self, input, metric)
    }

    fn distance_transform_edt(&self, input: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        distance_transform_edt_impl(self, input)
    }
}
