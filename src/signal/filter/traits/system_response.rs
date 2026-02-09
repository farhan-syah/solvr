//! System response traits.
//!
//! Provides algorithms for computing impulse and step responses of digital filters.

use crate::signal::filter::types::TransferFunction;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// System response algorithms.
///
/// All backends implementing system responses MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait SystemResponseAlgorithms<R: Runtime> {
    /// Compute the impulse response of a digital filter.
    ///
    /// # Algorithm
    ///
    /// Applies the filter to an impulse input: x = [1, 0, 0, 0, ...]
    /// The output is the filter's impulse response `h[n]`.
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function of the filter
    /// * `n` - Number of output samples
    ///
    /// # Returns
    ///
    /// [`ImpulseResponse`] containing:
    /// - `t`: Time indices `[0, 1, 2, ..., n-1]`
    /// - `y`: Impulse response samples
    fn impulse(
        &self,
        tf: &TransferFunction<R>,
        n: usize,
        device: &R::Device,
    ) -> Result<ImpulseResponse<R>>;

    /// Compute the step response of a digital filter.
    ///
    /// # Algorithm
    ///
    /// Applies the filter to a step input: x = [1, 1, 1, 1, ...]
    /// The output is the filter's step response.
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function of the filter
    /// * `n` - Number of output samples
    ///
    /// # Returns
    ///
    /// [`StepResponse`] containing:
    /// - `t`: Time indices `[0, 1, 2, ..., n-1]`
    /// - `y`: Step response samples
    fn step(
        &self,
        tf: &TransferFunction<R>,
        n: usize,
        device: &R::Device,
    ) -> Result<StepResponse<R>>;
}

/// Result from impulse response computation.
#[derive(Debug, Clone)]
pub struct ImpulseResponse<R: Runtime> {
    /// Time indices (sample numbers).
    pub t: Tensor<R>,
    /// Impulse response samples.
    pub y: Tensor<R>,
}

/// Result from step response computation.
#[derive(Debug, Clone)]
pub struct StepResponse<R: Runtime> {
    /// Time indices (sample numbers).
    pub t: Tensor<R>,
    /// Step response samples.
    pub y: Tensor<R>,
}
