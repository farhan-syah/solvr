//! Linear Programming algorithms.
//!
//! This module provides linear programming solvers including:
//! - `linprog` - Simplex method for linear programming
//! - `milp` - Mixed-integer linear programming via branch-and-bound
//!
//! # Runtime-Generic Architecture
//!
//! All operations are implemented generically over numr's `Runtime` trait.
//! The same code works on CPU, CUDA, and WebGPU backends with **zero duplication**.
//!
//! ```text
//! linprog/
//! ├── mod.rs              # Exports only
//! ├── traits/             # Trait definitions
//! ├── impl_generic/       # Generic implementations
//! ├── cpu/                # CPU implementations
//! ├── cuda/               # CUDA implementations
//! └── wgpu/               # WebGPU implementations
//! ```

mod traits;
mod impl_generic;
mod cpu;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "wgpu")]
mod wgpu;

pub use traits::{
    LinProgAlgorithms, LinProgOptions, LinProgTensorConstraints, LinProgTensorResult,
    MilpAlgorithms, MilpOptions, MilpTensorResult,
};
