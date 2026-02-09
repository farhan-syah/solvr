//! Semidefinite Programming (SDP) trait.
//!
//! Solves: min trace(C*X)
//!         s.t. trace(A_i*X) = b_i  for each constraint
//!              X >= 0 (positive semidefinite)

use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;

/// Options for SDP solver.
#[derive(Debug, Clone)]
pub struct SdpOptions {
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl Default for SdpOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-8,
        }
    }
}

/// Result of SDP solver.
#[derive(Debug, Clone)]
pub struct SdpResult<R: Runtime> {
    /// Solution matrix X `[n, n]`.
    pub x: Tensor<R>,
    /// Optimal objective value.
    pub fun: f64,
    /// Dual variables.
    pub dual: Tensor<R>,
    /// Number of iterations.
    pub iterations: usize,
    /// Whether converged.
    pub converged: bool,
}

/// Trait for SDP algorithms.
pub trait SdpAlgorithms<R: Runtime> {
    /// Solve a semidefinite program.
    ///
    /// min  trace(C*X)
    /// s.t. trace(A_i*X) = b_i
    ///      X >= 0  (positive semidefinite)
    ///
    /// # Arguments
    ///
    /// * `c_mat` - Cost matrix `[n, n]`
    /// * `a_mats` - Constraint matrices, each `[n, n]`
    /// * `b_vec` - Constraint RHS values `[m]`
    /// * `options` - Solver options
    fn solve_sdp(
        &self,
        c_mat: &Tensor<R>,
        a_mats: &[Tensor<R>],
        b_vec: &Tensor<R>,
        options: &SdpOptions,
    ) -> OptimizeResult<SdpResult<R>>;
}
