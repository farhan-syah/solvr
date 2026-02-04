//! Generic implementations of filter algorithms.
//!
//! All filter operations are fully tensor-based - data stays on device
//! with no GPU->CPU->GPU roundtrips.
//!
//! Note: Some inherently sequential algorithms are implemented as CPU-only
//! in the cpu/ directory since GPU provides no benefit:
//! - SOS conversions (sos2tf, zpk2sos, sos2zpk): pole/zero pairing, quadratic root finding
//! - FIR design (firwin, firwin2, minimum_phase): design-time sequential algorithms
//! - System response (dlsim, state_space): per-sample state updates

mod bilinear;
mod conversions;
mod fir;
mod freq_transform;
mod freqs;
mod iir_wrapper;
mod prototypes;

// Re-export conversion functions
// Note: SOS conversions (tf2sos, zpk2sos, etc.) are CPU-only - see cpu/conversions.rs
pub use bilinear::bilinear_zpk_impl;
pub use conversions::{tf2zpk_impl, zpk2tf_impl};

// Frequency transformations - fully tensorized
pub use freq_transform::{lp2bp_zpk_impl, lp2bs_zpk_impl, lp2hp_zpk_impl, lp2lp_zpk_impl};

// Frequency response
pub use freqs::freqs_impl;

// IIR design wrapper
pub use iir_wrapper::iirfilter_impl;

// Analog prototypes
pub use prototypes::{
    besselap_impl, buttap_impl, butter_impl, cheb1ap_impl, cheb2ap_impl, cheby1_impl, cheby2_impl,
    design_iir_filter, ellip_impl, ellipap_impl,
};

// Note: FIR design functions (firwin, firwin2, minimum_phase) are CPU-only
// and implemented directly in cpu/fir_design.rs
