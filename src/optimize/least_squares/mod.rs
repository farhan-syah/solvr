//! Least squares optimization and curve fitting.
//!
//! This module provides methods for solving nonlinear least squares problems:
//! minimize ||f(x)||^2 = sum(f_i(x)^2)
//!
//! where f: R^n -> R^m is a vector-valued function (residuals).
//!
//! # Runtime-Generic Architecture
//!
//! All operations are implemented generically over numr's `Runtime` trait.
//! The same code works on CPU, CUDA, and WebGPU backends with **zero duplication**.
//!
//! ```text
//! least_squares/
//! ├── mod.rs                   # Exports only
//! ├── traits/
//! │   ├── mod.rs               # Exports only
//! │   └── least_squares.rs     # Trait definition + types
//! ├── impl_generic/
//! │   ├── mod.rs               # Exports only
//! │   ├── leastsq.rs           # Unbounded LM algorithm
//! │   └── bounded.rs           # Bounded LM algorithm
//! ├── cpu/
//! │   ├── mod.rs               # Exports only
//! │   └── least_squares.rs     # CPU impl
//! ├── cuda/
//! │   ├── mod.rs               # Exports only
//! │   └── least_squares.rs     # CUDA impl
//! └── wgpu/
//!     ├── mod.rs               # Exports only
//!     └── least_squares.rs     # WebGPU impl
//! ```

mod cpu;

#[cfg(feature = "cuda")]
mod cuda;

pub mod impl_generic;
pub mod traits;

#[cfg(feature = "wgpu")]
mod wgpu;

pub use traits::{LeastSquaresAlgorithms, LeastSquaresOptions, LeastSquaresTensorResult};
