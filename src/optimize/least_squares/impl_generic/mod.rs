//! Generic implementations of least squares algorithms.
use crate::DType;

mod bounded;
mod leastsq;

pub use bounded::least_squares_impl;
pub use leastsq::leastsq_impl;

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Result from tensor-based least squares optimization.
#[derive(Debug, Clone)]
pub struct TensorLeastSquaresResult<R: Runtime<DType = DType>> {
    /// The optimal parameters found
    pub x: Tensor<R>,
    /// Residual vector at solution
    pub residuals: Tensor<R>,
    /// Sum of squared residuals (cost)
    pub cost: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Number of function evaluations
    pub nfev: usize,
    /// Whether the method converged
    pub converged: bool,
}
