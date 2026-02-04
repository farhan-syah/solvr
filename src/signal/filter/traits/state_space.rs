//! State-space conversion traits.
//!
//! Provides algorithms for converting between transfer function and state-space
//! representations.

use crate::signal::filter::types::{StateSpace, TransferFunction};
use numr::error::Result;
use numr::runtime::Runtime;

/// State-space conversion algorithms.
///
/// All backends implementing state-space conversions MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait StateSpaceConversions<R: Runtime> {
    /// Convert transfer function to controllable canonical state-space form.
    ///
    /// # Algorithm
    ///
    /// For a transfer function H(z) = B(z) / A(z) with:
    /// - A(z) = a[0] + a[1]z^-1 + ... + a[n]z^-n  (a[0] = 1 after normalization)
    /// - B(z) = b[0] + b[1]z^-1 + ... + b[m]z^-m
    ///
    /// Constructs the controllable canonical form:
    /// ```text
    /// A = | 0  1  0  ...  0  |     B = | 0 |
    ///     | 0  0  1  ...  0  |         | 0 |
    ///     | :  :  :   ⋱   :  |         | : |
    ///     | 0  0  0  ...  1  |         | 0 |
    ///     |-aₙ -aₙ₋₁ ... -a₁|         | 1 |
    ///
    /// C = | b̃ₙ - bₙaₙ ... b̃₁ - b₁a₁ |     D = | b̃₀ |
    ///
    /// where b̃ᵢ = bᵢ (zero-padded if necessary)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function to convert
    ///
    /// # Returns
    ///
    /// State-space representation (A, B, C, D matrices).
    fn tf2ss(&self, tf: &TransferFunction<R>) -> Result<StateSpace<R>>;

    /// Convert state-space to transfer function.
    ///
    /// # Algorithm
    ///
    /// Computes H(s) = C * (sI - A)^{-1} * B + D
    ///
    /// For SISO systems, this gives:
    /// - Numerator: polynomial from C * adj(sI - A) * B + D * det(sI - A)
    /// - Denominator: characteristic polynomial det(sI - A)
    ///
    /// # Arguments
    ///
    /// * `ss` - State-space system to convert
    ///
    /// # Returns
    ///
    /// Transfer function representation (b, a coefficients).
    fn ss2tf(&self, ss: &StateSpace<R>) -> Result<TransferFunction<R>>;
}
