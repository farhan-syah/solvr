//! Least squares optimization trait and types.
use crate::DType;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Options for least squares optimization.
#[derive(Debug, Clone)]
pub struct LeastSquaresOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence (change in cost)
    pub f_tol: f64,
    /// Tolerance for convergence (change in parameters)
    pub x_tol: f64,
    /// Tolerance for convergence (gradient norm)
    pub g_tol: f64,
    /// Step size for finite difference Jacobian
    pub eps: f64,
}

impl Default for LeastSquaresOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            f_tol: 1e-8,
            x_tol: 1e-8,
            g_tol: 1e-8,
            eps: 1e-8,
        }
    }
}

/// Result from tensor-based least squares optimization.
#[derive(Debug, Clone)]
pub struct LeastSquaresTensorResult<R: Runtime<DType = DType>> {
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

/// Algorithmic contract for least squares optimization.
///
/// All backends implementing least squares MUST implement this trait using
/// the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait LeastSquaresAlgorithms<R: Runtime<DType = DType>> {
    /// Levenberg-Marquardt algorithm for nonlinear least squares.
    ///
    /// Minimizes ||f(x)||^2 where f: R^n -> R^m.
    ///
    /// # Arguments
    ///
    /// * `f` - Residual function returning tensor of residuals
    /// * `x0` - Initial parameter guess
    /// * `options` - Solver options
    ///
    /// # Returns
    ///
    /// Tensor result with optimal parameters
    fn leastsq<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &LeastSquaresOptions,
    ) -> Result<LeastSquaresTensorResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;

    /// Bounded Levenberg-Marquardt algorithm.
    ///
    /// Minimizes ||f(x)||^2 subject to lower <= x <= upper.
    ///
    /// # Arguments
    ///
    /// * `f` - Residual function
    /// * `x0` - Initial parameter guess
    /// * `bounds` - Optional (lower, upper) bounds tensors
    /// * `options` - Solver options
    fn least_squares<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        bounds: Option<(&Tensor<R>, &Tensor<R>)>,
        options: &LeastSquaresOptions,
    ) -> Result<LeastSquaresTensorResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = LeastSquaresOptions::default();
        assert_eq!(opts.max_iter, 100);
        assert!((opts.f_tol - 1e-8).abs() < 1e-12);
    }
}
