//! Shared types for trust region optimization methods.
//!
//! Trust region methods minimize f(x) by iteratively solving a subproblem
//! within a "trust region" ball ||p|| <= delta, then adjusting delta based
//! on how well the quadratic model predicted the actual reduction.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::optimize::minimize::TensorMinimizeResult;

/// Options for trust region optimization methods.
#[derive(Debug, Clone)]
pub struct TrustRegionOptions {
    /// Initial trust region radius
    pub initial_trust_radius: f64,
    /// Maximum trust region radius
    pub max_trust_radius: f64,
    /// Acceptance threshold eta: reject step if actual/predicted < eta
    pub eta: f64,
    /// Gradient norm tolerance for convergence
    pub gtol: f64,
    /// Maximum number of iterations
    pub max_iter: usize,
}

impl Default for TrustRegionOptions {
    fn default() -> Self {
        Self {
            initial_trust_radius: 1.0,
            max_trust_radius: 1000.0,
            eta: 0.15,
            gtol: 1e-8,
            max_iter: 200,
        }
    }
}

/// Result type for trust region optimization.
#[derive(Debug, Clone)]
pub struct TrustRegionResult<R: Runtime> {
    /// Solution vector
    pub x: Tensor<R>,
    /// Function value at solution
    pub fun: f64,
    /// Gradient at solution
    pub grad: Tensor<R>,
    /// Number of iterations
    pub iterations: usize,
    /// Whether the algorithm converged
    pub converged: bool,
    /// Final trust region radius
    pub trust_radius: f64,
    /// Number of function evaluations
    pub nfev: usize,
}

impl<R: Runtime> From<TrustRegionResult<R>> for TensorMinimizeResult<R> {
    fn from(result: TrustRegionResult<R>) -> Self {
        TensorMinimizeResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            nfev: result.nfev,
            converged: result.converged,
        }
    }
}
