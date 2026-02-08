//! WebGPU implementation of finite element PDE solvers.

use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::pde::error::PdeResult;
use crate::pde::impl_generic::{fem_1d_impl, fem_triangular_impl};
use crate::pde::traits::FiniteElementAlgorithms;
use crate::pde::types::{BoundarySpec, FdmOptions, FemResult};

impl FiniteElementAlgorithms<WgpuRuntime> for WgpuClient {
    fn fem_1d(
        &self,
        f_rhs: &Tensor<WgpuRuntime>,
        x_nodes: &Tensor<WgpuRuntime>,
        boundary: &[BoundarySpec<WgpuRuntime>],
        options: &FdmOptions,
    ) -> PdeResult<FemResult<WgpuRuntime>> {
        fem_1d_impl(self, f_rhs, x_nodes, boundary, options)
    }

    fn fem_triangular(
        &self,
        f_rhs: &Tensor<WgpuRuntime>,
        nodes: &Tensor<WgpuRuntime>,
        elements: &Tensor<WgpuRuntime>,
        boundary_nodes: &Tensor<WgpuRuntime>,
        boundary_values: &Tensor<WgpuRuntime>,
        options: &FdmOptions,
    ) -> PdeResult<FemResult<WgpuRuntime>> {
        fem_triangular_impl(
            self,
            f_rhs,
            nodes,
            elements,
            boundary_nodes,
            boundary_values,
            options,
        )
    }
}
