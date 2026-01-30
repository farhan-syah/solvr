//! Numerical quadrature (integration) methods.
//!
//! This module provides methods for numerically computing definite integrals.
//!
//! # Available Methods
//!
//! | Method | Use Case | Accuracy |
//! |--------|----------|----------|
//! | [`trapezoid`] | Sampled data, uniform spacing | O(h²) |
//! | [`simpson`] | Sampled data, uniform spacing | O(h⁴) |
//! | [`fixed_quad`] | Smooth functions | Exact for polynomials up to degree 2n-1 |
//! | [`quad`] | General functions | Adaptive to specified tolerance |
//! | [`romberg`] | Smooth functions | High precision via extrapolation |
//!
//! # Choosing a Method
//!
//! - **Sampled data**: Use [`trapezoid`] or [`simpson`]
//! - **Smooth functions**: Use [`fixed_quad`] for efficiency or [`quad`] for reliability
//! - **High precision needed**: Use [`romberg`] or [`quad`] with tight tolerances
//! - **Functions with singularities**: Use [`quad`] (handles endpoint singularities)

mod adaptive;
mod gauss;
mod simpson;
mod trapezoid;

// Re-export all public items
pub use adaptive::{QuadOptions, QuadResult, RombergOptions, quad, romberg};
pub use gauss::{GaussLegendreQuadrature, fixed_quad};
pub use simpson::{simpson, simpson_func};
pub use trapezoid::{cumulative_trapezoid, trapezoid, trapezoid_uniform};
