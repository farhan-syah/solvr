//! Shared types for PDE solvers.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

// ============================================================================
// Grid Types
// ============================================================================

/// 2D uniform rectilinear grid.
#[derive(Debug, Clone)]
pub struct Grid2D {
    /// Number of grid points in x direction.
    pub nx: usize,
    /// Number of grid points in y direction.
    pub ny: usize,
    /// Grid spacing in x direction.
    pub dx: f64,
    /// Grid spacing in y direction.
    pub dy: f64,
    /// Domain extent in x: [x_min, x_max].
    pub x_range: [f64; 2],
    /// Domain extent in y: [y_min, y_max].
    pub y_range: [f64; 2],
}

/// 3D uniform rectilinear grid.
#[derive(Debug, Clone)]
pub struct Grid3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
    pub x_range: [f64; 2],
    pub y_range: [f64; 2],
    pub z_range: [f64; 2],
}

// ============================================================================
// Boundary Conditions
// ============================================================================

/// Boundary condition type.
#[derive(Debug, Clone)]
pub enum BoundaryCondition<R: Runtime> {
    /// Fixed values at boundary (Dirichlet).
    Dirichlet(Tensor<R>),
    /// Fixed normal derivative at boundary (Neumann).
    Neumann(Tensor<R>),
    /// Periodic boundary (wraps around).
    Periodic,
}

/// Which side of the domain to apply a boundary condition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundarySide {
    Left,
    Right,
    Bottom,
    Top,
    Front,
    Back,
    All,
}

/// A boundary specification: side + condition.
#[derive(Debug, Clone)]
pub struct BoundarySpec<R: Runtime> {
    pub side: BoundarySide,
    pub condition: BoundaryCondition<R>,
}

// ============================================================================
// Solver Options
// ============================================================================

/// Sparse linear solver selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SparseSolver {
    /// Conjugate Gradient (symmetric positive definite).
    #[default]
    Cg,
    /// GMRES (general non-symmetric).
    Gmres,
    /// BiCGSTAB (general, lower memory than GMRES).
    BiCgStab,
}

/// Preconditioner selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Preconditioner {
    /// No preconditioning.
    #[default]
    None,
    /// Incomplete LU factorization (zero fill-in).
    Ilu0,
    /// Incomplete Cholesky (zero fill-in, for SPD systems).
    Ic0,
}

/// Options for finite difference method solvers.
#[derive(Debug, Clone)]
pub struct FdmOptions {
    pub solver: SparseSolver,
    pub preconditioner: Preconditioner,
    pub max_iter: usize,
    pub tolerance: f64,
}

impl Default for FdmOptions {
    fn default() -> Self {
        Self {
            solver: SparseSolver::Cg,
            preconditioner: Preconditioner::None,
            max_iter: 10000,
            tolerance: 1e-10,
        }
    }
}

/// Options for time-dependent PDE solvers.
#[derive(Debug, Clone)]
pub struct TimeDependentOptions {
    /// Time span [t_start, t_end].
    pub t_span: [f64; 2],
    /// Time step size. None = adaptive.
    pub dt: Option<f64>,
    /// Save solution every N steps (0 = only final).
    pub save_every: usize,
}

// ============================================================================
// Result Types
// ============================================================================

/// Result of a finite difference steady-state solve.
#[derive(Debug, Clone)]
pub struct FdmResult<R: Runtime> {
    /// Solution field (reshaped to grid dimensions).
    pub solution: Tensor<R>,
    /// Number of iterations used by the iterative solver.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: f64,
}

/// Result of a time-dependent PDE solve.
#[derive(Debug, Clone)]
pub struct TimeResult<R: Runtime> {
    /// Time points at which solutions were saved.
    pub times: Tensor<R>,
    /// Solutions at saved timesteps.
    pub solutions: Vec<Tensor<R>>,
}

/// Result of a finite element solve.
#[derive(Debug, Clone)]
pub struct FemResult<R: Runtime> {
    /// Solution at nodes.
    pub solution: Tensor<R>,
    /// Node coordinates.
    pub nodes: Tensor<R>,
    /// Number of solver iterations.
    pub iterations: usize,
    /// Final residual norm.
    pub residual_norm: f64,
}

/// Result of a spectral method solve.
#[derive(Debug, Clone)]
pub struct SpectralResult<R: Runtime> {
    /// Solution at collocation nodes.
    pub solution: Tensor<R>,
    /// Collocation node coordinates.
    pub nodes: Tensor<R>,
}
