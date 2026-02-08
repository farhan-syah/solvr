//! CUDA implementation of halfspace intersection.

use crate::spatial::impl_generic::halfspace_intersection_impl;
use crate::spatial::traits::halfspace_intersection::{
    HalfspaceIntersection, HalfspaceIntersectionAlgorithms,
};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl HalfspaceIntersectionAlgorithms<CudaRuntime> for CudaClient {
    fn halfspace_intersection(
        &self,
        halfspaces: &Tensor<CudaRuntime>,
        interior_point: &Tensor<CudaRuntime>,
    ) -> Result<HalfspaceIntersection<CudaRuntime>> {
        halfspace_intersection_impl(self, halfspaces, interior_point)
    }
}
