//! Newton's method trait for systems of nonlinear equations.
use crate::DType;

use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Newton's method for systems of nonlinear equations.
///
/// Uses finite differences to approximate the Jacobian.
pub trait NewtonSystemAlgorithms<R: Runtime<DType = DType>> {
    /// Solve F(x) = 0 using Newton's method.
    ///
    /// # Arguments
    /// * `f` - The function F: R^n -> R^n
    /// * `x0` - Initial guess
    /// * `options` - Solver options
    fn newton_system<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;
}
