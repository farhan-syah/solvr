//! Generic implementations of filter algorithms.
//!
//! All filter operations are fully tensor-based - data stays on device
//! with no GPU->CPU->GPU roundtrips.
//!
//! Note: Some inherently sequential algorithms (system_response, dlsim, state_space)
//! are implemented as CPU-only in the cpu/ directory since GPU provides no benefit.

mod bilinear;
mod conversions;
mod fir;
mod freq_transform;
mod freqs;
mod iir_wrapper;
mod prototypes;

// Re-export for backend implementations
pub use bilinear::bilinear_zpk_impl;
pub use conversions::{
    sos2tf_impl, sos2zpk_impl, tf2sos_impl, tf2zpk_impl, zpk2sos_impl, zpk2tf_impl,
};
pub use fir::{firwin_impl, firwin2_impl, minimum_phase_impl};
pub use freq_transform::{lp2bp_zpk_impl, lp2bs_zpk_impl, lp2hp_zpk_impl, lp2lp_zpk_impl};
pub use freqs::freqs_impl;
pub use iir_wrapper::iirfilter_impl;
pub use prototypes::{
    besselap_impl, buttap_impl, butter_impl, cheb1ap_impl, cheb2ap_impl, cheby1_impl, cheby2_impl,
    design_iir_filter, ellip_impl, ellipap_impl,
};
