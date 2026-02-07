//! Conic programming algorithms.
//!
//! Provides SOCP and SDP solvers.

pub mod cpu;
pub mod impl_generic;
pub mod traits;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use traits::*;
