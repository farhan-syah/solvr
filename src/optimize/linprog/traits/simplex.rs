//! Linear programming trait and types.
use crate::DType;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Options for linear programming solvers.
#[derive(Debug, Clone)]
pub struct LinProgOptions {
    /// Maximum number of iterations.
    pub max_iter: usize,
    /// Tolerance for optimality.
    pub tol: f64,
    /// Whether to presolve (remove redundant constraints).
    pub presolve: bool,
}

impl Default for LinProgOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-9,
            presolve: true,
        }
    }
}

/// Tensor-based linear constraints.
#[derive(Debug, Clone)]
pub struct LinProgTensorConstraints<R: Runtime<DType = DType>> {
    /// Inequality constraint matrix (A_ub * x <= b_ub)
    pub a_ub: Option<Tensor<R>>,
    /// Inequality constraint bounds
    pub b_ub: Option<Tensor<R>>,
    /// Equality constraint matrix (A_eq * x == b_eq)
    pub a_eq: Option<Tensor<R>>,
    /// Equality constraint bounds
    pub b_eq: Option<Tensor<R>>,
    /// Variable lower bounds
    pub lower_bounds: Option<Tensor<R>>,
    /// Variable upper bounds
    pub upper_bounds: Option<Tensor<R>>,
}

/// Result from tensor-based linear programming.
#[derive(Debug, Clone)]
pub struct LinProgTensorResult<R: Runtime<DType = DType>> {
    /// Optimal solution vector
    pub x: Tensor<R>,
    /// Optimal objective value
    pub fun: f64,
    /// Whether optimization succeeded
    pub success: bool,
    /// Number of iterations performed
    pub nit: usize,
    /// Status message
    pub message: String,
    /// Slack variables for inequality constraints
    pub slack: Tensor<R>,
}

/// Algorithmic contract for linear programming operations.
///
/// All backends implementing linear programming MUST implement this trait using
/// the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait LinProgAlgorithms<R: Runtime<DType = DType>> {
    /// Solve a linear programming problem using the Simplex method.
    ///
    /// Minimize: c^T * x
    /// Subject to:
    ///   A_ub * x <= b_ub
    ///   A_eq * x == b_eq
    ///   lower_bounds <= x <= upper_bounds
    fn linprog(
        &self,
        c: &Tensor<R>,
        constraints: &LinProgTensorConstraints<R>,
        options: &LinProgOptions,
    ) -> Result<LinProgTensorResult<R>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = LinProgOptions::default();
        assert_eq!(opts.max_iter, 1000);
        assert!((opts.tol - 1e-9).abs() < 1e-12);
    }
}
