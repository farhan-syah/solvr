//! Unified IIR filter design wrapper traits.
//!
//! Provides a single interface to design IIR filters using various methods.

// Allow many arguments for filter design functions that match scipy's signature
#![allow(clippy::too_many_arguments)]
// Allow large enum variant size difference (ZpkFilter is larger than tf/sos)
#![allow(clippy::large_enum_variant)]

use crate::signal::filter::traits::iir_design::IirDesignResult;
use crate::signal::filter::types::{FilterOutput, FilterType};
use numr::error::Result;
use numr::runtime::Runtime;

/// IIR filter design type selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IirDesignType {
    /// Butterworth filter (maximally flat magnitude).
    #[default]
    Butterworth,
    /// Chebyshev Type I (equiripple passband).
    Chebyshev1,
    /// Chebyshev Type II (equiripple stopband).
    Chebyshev2,
    /// Elliptic/Cauer (equiripple both bands, sharpest transition).
    Elliptic,
    /// Bessel-Thomson (maximally flat group delay).
    Bessel,
}

/// Unified IIR filter design algorithms.
///
/// Provides a single interface to design IIR filters of various types.
pub trait IirFilterAlgorithms<R: Runtime> {
    /// Design an IIR digital filter using the specified method.
    ///
    /// This is a unified interface that dispatches to the appropriate design
    /// function based on `design_type`.
    ///
    /// # Arguments
    ///
    /// * `order` - Filter order (number of poles)
    /// * `wn` - Critical frequency (normalized, 0 < Wn < 1 for digital)
    ///   - For lowpass/highpass: single frequency
    ///   - For bandpass/bandstop: [low, high] frequencies
    /// * `filter_type` - Type of filter (lowpass, highpass, bandpass, bandstop)
    /// * `design_type` - IIR design method (Butterworth, Chebyshev, etc.)
    /// * `rp` - Maximum ripple in passband (dB). Required for Chebyshev1 and Elliptic.
    /// * `rs` - Minimum attenuation in stopband (dB). Required for Chebyshev2 and Elliptic.
    /// * `output` - Output format (Ba, Zpk, or Sos)
    ///
    /// # Returns
    ///
    /// Filter in requested format wrapped in [`IirDesignResult`].
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::runtime::cpu::{CpuClient, CpuDevice};
    /// use solvr::signal::filter::{IirFilterAlgorithms, IirDesignType, FilterType, FilterOutput};
    /// # let device = CpuDevice::new();
    /// # let client = CpuClient::new(device.clone());
    /// // Design a 4th-order Chebyshev Type I lowpass with 1dB ripple
    /// let result = client.iirfilter(
    ///     4,
    ///     &[0.2],
    ///     FilterType::Lowpass,
    ///     IirDesignType::Chebyshev1,
    ///     Some(1.0),
    ///     None,
    ///     FilterOutput::Sos,
    ///     &device,
    /// )?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    fn iirfilter(
        &self,
        order: usize,
        wn: &[f64],
        filter_type: FilterType,
        design_type: IirDesignType,
        rp: Option<f64>,
        rs: Option<f64>,
        output: FilterOutput,
        device: &R::Device,
    ) -> Result<IirDesignResult<R>>;
}
