//! Wavelet transform algorithms.
//!
//! Provides discrete and continuous wavelet transforms.

mod cpu;
mod impl_generic;
mod traits;
mod types;

pub use traits::{CwtAlgorithms, CwtResult, DwtAlgorithms, DwtResult, WavedecResult};
pub use types::{Wavelet, WaveletFamily};
