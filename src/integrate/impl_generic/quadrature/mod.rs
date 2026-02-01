//! Tensor-based quadrature implementations.
//!
//! This module provides numerical integration algorithms using tensor operations.
//! All implementations are generic over `R: Runtime` for multi-backend support.

mod fixed_quad;
mod quad;
mod romberg;
mod simpson;
mod trapezoid;

pub use fixed_quad::fixed_quad_impl;
pub use quad::quad_impl;
pub use romberg::romberg_impl;
pub use simpson::simpson_impl;
pub use trapezoid::{cumulative_trapezoid_impl, trapezoid_impl, trapezoid_uniform_impl};
