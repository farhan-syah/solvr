//! Multivariate unconstrained minimization options.
//!
//! This module provides configuration types for minimization algorithms.
//! The actual implementations are in `impl_generic/` and use tensor operations.

/// Options for multivariate minimization.
#[derive(Debug, Clone)]
pub struct MinimizeOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence (function value change)
    pub f_tol: f64,
    /// Tolerance for convergence (argument change)
    pub x_tol: f64,
    /// Tolerance for gradient norm (gradient-based methods)
    pub g_tol: f64,
    /// Step size for finite difference gradient approximation
    pub eps: f64,
}

impl Default for MinimizeOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            f_tol: 1e-8,
            x_tol: 1e-8,
            g_tol: 1e-8,
            eps: 1e-8,
        }
    }
}
