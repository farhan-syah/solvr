//! Powell hybrid method trait for nonlinear systems.
use crate::DType;

use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Powell hybrid method for systems of nonlinear equations.
///
/// Combines Newton steps with dogleg trust region fallback.
/// Uses finite-difference Jacobian with Broyden rank-1 updates
/// between full Jacobian recalculations.
pub trait PowellHybridAlgorithms<R: Runtime<DType = DType>> {
    /// Solve F(x) = 0 using Powell's hybrid method.
    ///
    /// # Arguments
    /// * `f` - The system F: R^n -> R^n
    /// * `x0` - Initial guess
    /// * `options` - Solver options
    fn powell_hybrid<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &RootOptions,
    ) -> Result<RootTensorResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;
}
