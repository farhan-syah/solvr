//! Trust-region constrained optimization algorithm trait.
use crate::DType;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;

use super::types::{Bounds, ConstrainedOptions, ConstrainedResult, Constraint};

/// Trait for trust-region constrained optimization.
pub trait TrustConstrAlgorithms<R: Runtime<DType = DType>> {
    /// Trust-region constrained optimization (trust-constr).
    ///
    /// Two-mode algorithm:
    /// - Equality-only: Byrd-Omojokun trust-region SQP
    /// - With inequalities: Interior point with log-barrier
    ///
    /// # Arguments
    ///
    /// * `f` - Objective function f: R^n -> R
    /// * `x0` - Initial guess
    /// * `constraints` - Slice of nonlinear constraints
    /// * `bounds` - Variable bounds (optional lower/upper)
    /// * `options` - Algorithm options
    fn trust_constr<F>(
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
