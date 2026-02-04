//! Digital filter design and representation.
//!
//! This module provides comprehensive digital filter functionality:
//!
//! # Filter Representations
//!
//! - [`TransferFunction`]: Numerator/denominator polynomials (b, a)
//! - [`ZpkFilter`]: Zeros, poles, and gain
//! - [`SosFilter`]: Second-order sections (cascaded biquads)
//!
//! # IIR Filter Design
//!
//! Classic IIR filter design using bilinear transform:
//! - [`butter`](IirDesignAlgorithms::butter) - Butterworth (maximally flat)
//! - [`cheby1`](IirDesignAlgorithms::cheby1) - Chebyshev Type I (passband ripple)
//! - [`cheby2`](IirDesignAlgorithms::cheby2) - Chebyshev Type II (stopband ripple)
//! - [`ellip`](IirDesignAlgorithms::ellip) - Elliptic (sharpest transition)
//! - [`bessel`](IirDesignAlgorithms::bessel) - Bessel-Thomson (linear phase)
//!
//! # FIR Filter Design
//!
//! - [`firwin`](FirDesignAlgorithms::firwin) - Windowed sinc method
//! - [`firwin2`](FirDesignAlgorithms::firwin2) - Frequency sampling method
//!
//! # Conversions
//!
//! Convert between representations via [`FilterConversions`]:
//! - `tf2zpk`, `zpk2tf` - Transfer function ↔ ZPK
//! - `tf2sos`, `sos2tf` - Transfer function ↔ SOS
//! - `zpk2sos`, `sos2zpk` - ZPK ↔ SOS
//!
//! # Example
//!
//! ```ignore
//! use solvr::signal::filter::{IirDesignAlgorithms, FilterType, FilterOutput};
//! use numr::runtime::cpu::{CpuClient, CpuDevice};
//!
//! let device = CpuDevice::new();
//! let client = CpuClient::new(device.clone());
//!
//! // Design a 4th-order Butterworth lowpass filter at 0.2 * Nyquist
//! let result = client.butter(4, &[0.2], FilterType::Lowpass, FilterOutput::Sos, &device)?;
//! let sos = result.as_sos().unwrap();
//! ```

pub mod impl_generic;
pub mod traits;
pub mod types;

mod cpu;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "wgpu")]
mod wgpu;

// Re-export types
pub use types::{
    AnalogPrototype, FilterOutput, FilterType, SosFilter, TransferFunction, ZpkFilter,
};

// Re-export traits
pub use traits::conversions::{FilterConversions, SosPairing};
pub use traits::fir_design::{FirDesignAlgorithms, FirWindow};
pub use traits::iir_design::{BesselNorm, IirDesignAlgorithms, IirDesignResult};
