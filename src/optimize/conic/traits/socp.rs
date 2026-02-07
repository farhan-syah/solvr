//! Second-Order Cone Programming (SOCP) trait.
//!
//! Solves: min c'*x
//!         s.t. ||A_i*x + b_i|| <= c_i'*x + d_i  for each cone constraint

use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;

/// A second-order cone constraint: ||A*x + b|| <= c'*x + d
#[derive(Debug, Clone)]
pub struct SocConstraint<R: Runtime> {
    /// Matrix A [m_i, n]
    pub a: Tensor<R>,
    /// Vector b [m_i]
    pub b: Tensor<R>,
    /// Vector c [n]
    pub c: Tensor<R>,
    /// Scalar d
    pub d: f64,
}

/// Options for SOCP solver.
#[derive(Debug, Clone)]
pub struct SocpOptions {
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance.
    pub tol: f64,
}

impl Default for SocpOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-8,
        }
    }
}

/// Result of SOCP solver.
#[derive(Debug, Clone)]
pub struct SocpResult<R: Runtime> {
    /// Solution vector.
    pub x: Tensor<R>,
    /// Optimal objective value.
    pub fun: f64,
    /// Number of iterations.
    pub iterations: usize,
    /// Whether converged.
    pub converged: bool,
}

/// Trait for SOCP algorithms.
pub trait SocpAlgorithms<R: Runtime> {
    /// Solve a second-order cone program.
    ///
    /// min  c'*x
    /// s.t. ||A_i*x + b_i|| <= c_i'*x + d_i  for each cone
    fn solve_socp(
        &self,
        c: &Tensor<R>,
        constraints: &[SocConstraint<R>],
        options: &SocpOptions,
    ) -> OptimizeResult<SocpResult<R>>;
}
