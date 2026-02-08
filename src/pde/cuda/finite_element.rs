//! CUDA implementation of finite element PDE solvers.

use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::pde::error::PdeResult;
use crate::pde::impl_generic::{fem_1d_impl, fem_triangular_impl};
use crate::pde::traits::FiniteElementAlgorithms;
use crate::pde::types::{BoundarySpec, FdmOptions, FemResult};

impl FiniteElementAlgorithms<CudaRuntime> for CudaClient {
    fn fem_1d(
        &self,
        f_rhs: &Tensor<CudaRuntime>,
        x_nodes: &Tensor<CudaRuntime>,
        boundary: &[BoundarySpec<CudaRuntime>],
        options: &FdmOptions,
    ) -> PdeResult<FemResult<CudaRuntime>> {
        fem_1d_impl(self, f_rhs, x_nodes, boundary, options)
    }

    fn fem_triangular(
        &self,
        f_rhs: &Tensor<CudaRuntime>,
        nodes: &Tensor<CudaRuntime>,
        elements: &Tensor<CudaRuntime>,
        boundary_nodes: &Tensor<CudaRuntime>,
        boundary_values: &Tensor<CudaRuntime>,
        options: &FdmOptions,
    ) -> PdeResult<FemResult<CudaRuntime>> {
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
