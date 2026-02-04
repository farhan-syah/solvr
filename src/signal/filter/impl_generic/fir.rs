//! FIR filter design implementations.
//!
//! Note: FIR filter design (firwin, firwin2, minimum_phase) is CPU-only because:
//! 1. These are design-time operations (run once), not runtime (per-sample)
//! 2. FIR tap counts are typically 31-255 (tiny data)
//! 3. Algorithms involve sequential operations (sinc computation, interpolation)
//!
//! The full implementations are in cpu/fir_design.rs.

// This module is intentionally minimal.
// FIR design functions are CPU-only and implemented directly in cpu/fir_design.rs.
