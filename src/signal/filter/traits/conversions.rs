//! Filter representation conversion traits.
//!
//! Provides conversions between transfer function (b, a), zero-pole-gain (zpk),
//! and second-order sections (sos) representations.
use crate::DType;

use crate::signal::filter::types::{SosFilter, TransferFunction, ZpkFilter};
use numr::error::Result;
use numr::runtime::Runtime;

/// Filter representation conversions.
///
/// All backends implementing filter conversions MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait FilterConversions<R: Runtime<DType = DType>> {
    /// Convert transfer function to zeros, poles, and gain.
    ///
    /// # Algorithm
    ///
    /// 1. Find roots of numerator polynomial → zeros
    /// 2. Find roots of denominator polynomial → poles
    /// 3. Compute gain as ratio of leading coefficients
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function (b, a coefficients)
    ///
    /// # Returns
    ///
    /// ZPK representation with zeros, poles, and gain.
    fn tf2zpk(&self, tf: &TransferFunction<R>) -> Result<ZpkFilter<R>>;

    /// Convert zeros, poles, and gain to transfer function.
    ///
    /// # Algorithm
    ///
    /// 1. Build numerator from roots (zeros) using polyfromroots
    /// 2. Build denominator from roots (poles) using polyfromroots
    /// 3. Scale numerator by gain
    ///
    /// # Arguments
    ///
    /// * `zpk` - Zero-pole-gain representation
    ///
    /// # Returns
    ///
    /// Transfer function with (b, a) coefficients.
    fn zpk2tf(&self, zpk: &ZpkFilter<R>) -> Result<TransferFunction<R>>;

    /// Convert transfer function to second-order sections.
    ///
    /// # Algorithm
    ///
    /// 1. Convert TF to ZPK via tf2zpk
    /// 2. Convert ZPK to SOS via zpk2sos
    ///
    /// # Arguments
    ///
    /// * `tf` - Transfer function
    /// * `pairing` - How to pair poles/zeros (default: nearest)
    ///
    /// # Returns
    ///
    /// Second-order sections representation.
    fn tf2sos(&self, tf: &TransferFunction<R>, pairing: Option<SosPairing>)
    -> Result<SosFilter<R>>;

    /// Convert second-order sections to transfer function.
    ///
    /// # Algorithm
    ///
    /// Multiply all section transfer functions together:
    /// ```text
    /// H(z) = H_1(z) * H_2(z) * ... * H_n(z)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `sos` - Second-order sections
    ///
    /// # Returns
    ///
    /// Transfer function (b, a).
    fn sos2tf(&self, sos: &SosFilter<R>) -> Result<TransferFunction<R>>;

    /// Convert zeros, poles, gain to second-order sections.
    ///
    /// # Algorithm
    ///
    /// 1. Pair complex conjugate poles/zeros
    /// 2. Match real poles/zeros or pair with complex
    /// 3. Form biquad sections from each pair
    /// 4. Order sections for numerical stability
    ///
    /// # Pairing Strategy
    ///
    /// - `Nearest`: Pair poles with nearest zeros (default, most stable)
    /// - `MinimumPhase`: Place zeros inside unit circle first
    /// - `KeepOdd`: Don't pair the odd pole/zero
    ///
    /// # Arguments
    ///
    /// * `zpk` - Zero-pole-gain representation
    /// * `pairing` - Pairing strategy (default: Nearest)
    ///
    /// # Returns
    ///
    /// Second-order sections [n_sections, 6].
    fn zpk2sos(&self, zpk: &ZpkFilter<R>, pairing: Option<SosPairing>) -> Result<SosFilter<R>>;

    /// Convert second-order sections to zeros, poles, gain.
    ///
    /// # Algorithm
    ///
    /// For each section, find the 2 poles and 2 zeros of the biquad,
    /// then concatenate all.
    ///
    /// # Arguments
    ///
    /// * `sos` - Second-order sections
    ///
    /// # Returns
    ///
    /// ZPK representation.
    fn sos2zpk(&self, sos: &SosFilter<R>) -> Result<ZpkFilter<R>>;
}

/// Strategy for pairing poles and zeros when converting to SOS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SosPairing {
    /// Pair poles with nearest zeros (most numerically stable).
    #[default]
    Nearest,
    /// Pair to achieve minimum phase response.
    MinimumPhase,
    /// Keep odd pole/zero unpaired (for odd-order filters).
    KeepOdd,
}
