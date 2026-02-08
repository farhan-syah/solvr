//! Finite element method traits for PDE solving.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::pde::error::PdeResult;
use crate::pde::types::{BoundarySpec, FdmOptions, FemResult};

/// Finite element method algorithms for PDE solving.
pub trait FiniteElementAlgorithms<R: Runtime> {
    /// 1D FEM: -d/dx(du/dx) + q(x)u = f(x) on interval defined by x_nodes.
    ///
    /// Uses linear hat basis functions, assembles tridiagonal stiffness matrix,
    /// solves with banded solver.
    fn fem_1d(
        &self,
        f_rhs: &Tensor<R>,
        x_nodes: &Tensor<R>,
        boundary: &[BoundarySpec<R>],
        options: &FdmOptions,
    ) -> PdeResult<FemResult<R>>;

    /// 2D FEM on triangular mesh: -nabla^2 u = f.
    ///
    /// Uses linear triangular basis functions, sparse global assembly via
    /// batched tensor operations, solves with CG + preconditioner.
    fn fem_triangular(
        &self,
        f_rhs: &Tensor<R>,
        nodes: &Tensor<R>,
        elements: &Tensor<R>,
        boundary_nodes: &Tensor<R>,
        boundary_values: &Tensor<R>,
        options: &FdmOptions,
    ) -> PdeResult<FemResult<R>>;
}
