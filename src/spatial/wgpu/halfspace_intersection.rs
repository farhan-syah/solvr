//! WebGPU implementation of halfspace intersection.

use crate::spatial::impl_generic::halfspace_intersection_impl;
use crate::spatial::traits::halfspace_intersection::{
    HalfspaceIntersection, HalfspaceIntersectionAlgorithms,
};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl HalfspaceIntersectionAlgorithms<WgpuRuntime> for WgpuClient {
    fn halfspace_intersection(
        &self,
        halfspaces: &Tensor<WgpuRuntime>,
        interior_point: &Tensor<WgpuRuntime>,
    ) -> Result<HalfspaceIntersection<WgpuRuntime>> {
        halfspace_intersection_impl(self, halfspaces, interior_point)
    }
}
