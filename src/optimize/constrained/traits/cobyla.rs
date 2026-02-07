//! COBYLA (Constrained Optimization BY Linear Approximation) algorithm trait.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;

use super::types::{Bounds, ConstrainedOptions, ConstrainedResult, Constraint};

/// Trait for COBYLA constrained optimization.
pub trait CobylaAlgorithms<R: Runtime> {
    /// Constrained Optimization BY Linear Approximation (COBYLA).
    ///
    /// Derivative-free constrained optimization using linear approximations
    /// within a trust region framework.
    ///
    /// # Arguments
    ///
    /// * `f` - Objective function f: R^n -> R
    /// * `x0` - Initial guess
    /// * `constraints` - Slice of nonlinear constraints
    /// * `bounds` - Variable bounds (optional lower/upper)
    /// * `options` - Algorithm options
    fn cobyla<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        constraints: &[Constraint<'_, R>],
        bounds: &Bounds<R>,
        options: &ConstrainedOptions,
    ) -> OptimizeResult<ConstrainedResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;
}
