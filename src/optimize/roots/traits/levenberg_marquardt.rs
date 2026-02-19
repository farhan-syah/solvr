//! Levenberg-Marquardt algorithm trait for systems of nonlinear equations.
use crate::DType;

use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Levenberg-Marquardt algorithm for systems of nonlinear equations.
///
/// A damped Newton method that interpolates between Newton's method and
/// gradient descent. More robust when initial guess is far from solution.
pub trait LevenbergMarquardtAlgorithms<R: Runtime<DType = DType>> {
    /// Solve F(x) = 0 using Levenberg-Marquardt algorithm.
    ///
    /// # Arguments
    /// * `f` - The function F: R^n -> R^n
    /// * `x0` - Initial guess
    /// * `options` - Solver options
    fn levenberg_marquardt<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;
}
