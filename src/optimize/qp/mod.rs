//! Quadratic programming algorithms.
//!
//! Provides active set and interior point methods for solving QPs.

pub mod cpu;
pub mod impl_generic;
pub mod traits;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use traits::*;
