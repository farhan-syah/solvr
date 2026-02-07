//! Constrained optimization algorithms.
//!
//! Provides SLSQP, COBYLA, and trust-constr methods for optimization
//! with equality/inequality constraints and variable bounds.

pub mod cpu;
pub mod impl_generic;
pub mod traits;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

pub use traits::*;
