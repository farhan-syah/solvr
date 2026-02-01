//! Broyden's method trait for systems of nonlinear equations.

use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Broyden's method (rank-1 update) for systems of nonlinear equations.
///
/// A quasi-Newton method that approximates the Jacobian using rank-1 updates.
pub trait Broyden1Algorithms<R: Runtime> {
    /// Solve F(x) = 0 using Broyden's method.
    ///
    /// # Arguments
    /// * `f` - The function F: R^n -> R^n
    /// * `x0` - Initial guess
    /// * `options` - Solver options
    fn broyden1<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;
}
