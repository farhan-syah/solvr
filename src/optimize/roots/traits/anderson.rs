//! Anderson mixing trait for fixed-point iteration acceleration.
use crate::DType;

use crate::optimize::roots::{RootOptions, RootTensorResult};
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Options specific to Anderson mixing.
#[derive(Debug, Clone)]
pub struct AndersonOptions {
    /// Base root finding options
    pub base: RootOptions,
    /// Number of previous iterates to store (history depth)
    pub m: usize,
    /// Mixing parameter alpha (0 < alpha <= 1)
    pub alpha: f64,
}

impl Default for AndersonOptions {
    fn default() -> Self {
        Self {
            base: RootOptions::default(),
            m: 5,
            alpha: 1.0,
        }
    }
}

/// Anderson mixing for accelerating fixed-point iterations.
///
/// Given g(x) such that the fixed point x* = g(x*), Anderson mixing
/// finds x* by combining previous iterates using least-squares to
/// minimize the residual F(x) = g(x) - x.
pub trait AndersonAlgorithms<R: Runtime<DType = DType>> {
    /// Solve F(x) = g(x) - x = 0 using Anderson mixing.
    ///
    /// # Arguments
    /// * `g` - The fixed-point function g: R^n -> R^n
    /// * `x0` - Initial guess
    /// * `options` - Anderson mixing options
    fn anderson<G>(
        &self,
        g: G,
        x0: &Tensor<R>,
        options: &AndersonOptions,
    ) -> Result<RootTensorResult<R>>
    where
        G: Fn(&Tensor<R>) -> Result<Tensor<R>>;
}
