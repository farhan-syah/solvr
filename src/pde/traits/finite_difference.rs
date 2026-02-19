//! Finite difference method traits for PDE solving.
use crate::DType;

use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::pde::error::PdeResult;
use crate::pde::types::{
    BoundarySpec, FdmOptions, FdmResult, Grid2D, Grid3D, TimeDependentOptions, TimeResult,
};

/// Finite difference method algorithms for PDE solving.
///
/// Provides solvers for elliptic (Poisson), parabolic (heat), and hyperbolic
/// (wave) PDEs using sparse stencil discretization and iterative solvers.
#[allow(clippy::too_many_arguments)]
pub trait FiniteDifferenceAlgorithms<R: Runtime<DType = DType>> {
    /// Solve the Poisson equation: -nabla^2 u = f on a 2D grid.
    ///
    /// Assembles 5-point Laplacian stencil as sparse CSR matrix and solves
    /// with an iterative solver (CG for SPD systems).
    fn fdm_poisson(
        &self,
        f: &Tensor<R>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<R>],
        options: &FdmOptions,
    ) -> PdeResult<FdmResult<R>>;

    /// Solve the 2D heat equation: du/dt = alpha * nabla^2 u + source.
    ///
    /// Method of lines: spatial FDM discretization produces an ODE system,
    /// integrated in time using solvr's ODE solvers.
    fn fdm_heat_2d(
        &self,
        u0: &Tensor<R>,
        alpha: f64,
        source: Option<&Tensor<R>>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<R>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<R>>;

    /// Solve the 3D heat equation: du/dt = alpha * nabla^2 u + source.
    fn fdm_heat_3d(
        &self,
        u0: &Tensor<R>,
        alpha: f64,
        source: Option<&Tensor<R>>,
        grid: &Grid3D,
        boundary: &[BoundarySpec<R>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<R>>;

    /// Solve the 2D wave equation: d^2u/dt^2 = c^2 * nabla^2 u + source.
    ///
    /// Converts to first-order system [u, v] where v = du/dt, then uses
    /// ODE solver (Verlet/Leapfrog or RK45).
    fn fdm_wave(
        &self,
        u0: &Tensor<R>,
        v0: &Tensor<R>,
        c: f64,
        source: Option<&Tensor<R>>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<R>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<R>>;
}
