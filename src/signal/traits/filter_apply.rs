//! Filter application traits.
//!
//! Provides algorithms for applying digital filters to signals.

use crate::signal::filter::types::SosFilter;
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Filter application algorithms.
///
/// All backends implementing filter application MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait FilterApplicationAlgorithms<R: Runtime> {
    /// Apply an IIR or FIR filter using Direct Form II transposed.
    ///
    /// # Algorithm
    ///
    /// Direct Form II Transposed (most numerically stable):
    /// ```text
    /// y[n] = b[0]*x[n] + z[0]
    /// z[0] = b[1]*x[n] - a[1]*y[n] + z[1]
    /// z[1] = b[2]*x[n] - a[2]*y[n] + z[2]
    /// ...
    /// z[M-2] = b[M-1]*x[n] - a[M-1]*y[n]
    /// ```
    ///
    /// # Arguments
    ///
    /// * `b` - Numerator coefficients [M] in descending power order
    /// * `a` - Denominator coefficients [N] in descending power order (a[0] should be 1)
    /// * `x` - Input signal [..., samples]
    /// * `zi` - Initial filter state [max(M,N)-1], or None for zero state
    ///
    /// # Returns
    ///
    /// Filtered signal with same shape as input.
    fn lfilter(
        &self,
        b: &Tensor<R>,
        a: &Tensor<R>,
        x: &Tensor<R>,
        zi: Option<&Tensor<R>>,
    ) -> Result<LfilterResult<R>>;

    /// Zero-phase digital filtering (forward-backward filtering).
    ///
    /// # Algorithm
    ///
    /// 1. Apply filter forward: y1 = lfilter(b, a, x)
    /// 2. Reverse: y1_rev = reverse(y1)
    /// 3. Apply filter backward: y2 = lfilter(b, a, y1_rev)
    /// 4. Reverse: y = reverse(y2)
    ///
    /// This doubles the filter order but eliminates phase distortion.
    ///
    /// # Arguments
    ///
    /// * `b` - Numerator coefficients
    /// * `a` - Denominator coefficients
    /// * `x` - Input signal
    /// * `padtype` - Type of padding to use (default: Odd)
    /// * `padlen` - Number of samples to pad (default: 3 * max(len(a), len(b)))
    ///
    /// # Returns
    ///
    /// Zero-phase filtered signal.
    fn filtfilt(
        &self,
        b: &Tensor<R>,
        a: &Tensor<R>,
        x: &Tensor<R>,
        padtype: Option<PadType>,
        padlen: Option<usize>,
    ) -> Result<Tensor<R>>;

    /// Apply a filter in second-order sections form.
    ///
    /// # Algorithm
    ///
    /// For each biquad section, apply Direct Form II:
    /// ```text
    /// y[n] = b0*x[n] + z[0]
    /// z[0] = b1*x[n] - a1*y[n] + z[1]
    /// z[1] = b2*x[n] - a2*y[n]
    /// ```
    ///
    /// The output of each section becomes the input to the next.
    ///
    /// # Arguments
    ///
    /// * `sos` - Second-order sections [n_sections, 6]
    /// * `x` - Input signal
    /// * `zi` - Initial state [n_sections, 2], or None for zero state
    ///
    /// # Returns
    ///
    /// Filtered signal and final state.
    fn sosfilt(
        &self,
        sos: &SosFilter<R>,
        x: &Tensor<R>,
        zi: Option<&Tensor<R>>,
    ) -> Result<SosfiltResult<R>>;

    /// Zero-phase filtering using second-order sections.
    ///
    /// Combines sosfilt with forward-backward filtering for zero phase.
    fn sosfiltfilt(
        &self,
        sos: &SosFilter<R>,
        x: &Tensor<R>,
        padtype: Option<PadType>,
        padlen: Option<usize>,
    ) -> Result<Tensor<R>>;
}

/// Result from lfilter containing output and final state.
#[derive(Debug, Clone)]
pub struct LfilterResult<R: Runtime> {
    /// Filtered output signal.
    pub y: Tensor<R>,
    /// Final filter state (can be used as zi for next call).
    pub zf: Tensor<R>,
}

/// Result from sosfilt containing output and final state.
#[derive(Debug, Clone)]
pub struct SosfiltResult<R: Runtime> {
    /// Filtered output signal.
    pub y: Tensor<R>,
    /// Final filter state [n_sections, 2].
    pub zf: Tensor<R>,
}

/// Padding type for filtfilt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PadType {
    /// Odd extension (default): x[-n] = 2*x[0] - x[n]
    #[default]
    Odd,
    /// Even extension: x[-n] = x[n]
    Even,
    /// Constant extension: x[-n] = x[0]
    Constant,
    /// No padding.
    None,
}
