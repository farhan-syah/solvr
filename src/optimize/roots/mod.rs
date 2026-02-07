//! Multivariate root finding algorithms.
//!
//! This module provides methods for finding roots of systems of nonlinear equations.
//! Given F: R^n -> R^n, find x such that F(x) = 0.
//!
//! # Runtime-Generic Architecture
//!
//! All operations are implemented generically over numr's `Runtime` trait.
//! The same code works on CPU, CUDA, and WebGPU backends with **zero duplication**.
//!
//! ```text
//! roots/
//! ├── mod.rs                # Exports only (pub mod + pub use)
//! ├── traits/
//! │   ├── mod.rs            # Exports only
//! │   ├── newton.rs         # NewtonSystemAlgorithms trait
//! │   ├── broyden.rs        # Broyden1Algorithms trait
//! │   └── levenberg_marquardt.rs # LevenbergMarquardtAlgorithms trait
//! ├── impl_generic/
//! │   ├── mod.rs            # Exports + TensorRootResult
//! │   ├── newton.rs         # newton_system_impl()
//! │   ├── broyden.rs        # broyden1_impl()
//! │   └── levenberg_marquardt.rs # levenberg_marquardt_impl()
//! ├── cpu/
//! │   ├── mod.rs            # Exports only
//! │   ├── newton.rs         # CpuClient impl for NewtonSystemAlgorithms
//! │   ├── broyden.rs        # CpuClient impl for Broyden1Algorithms
//! │   └── levenberg_marquardt.rs # CpuClient impl for LevenbergMarquardtAlgorithms
//! ├── cuda/
//! │   ├── mod.rs            # Exports only (feature-gated)
//! │   ├── newton.rs         # CudaClient impl
//! │   ├── broyden.rs        # CudaClient impl
//! │   └── levenberg_marquardt.rs # CudaClient impl
//! └── wgpu/
//!     ├── mod.rs            # Exports only (feature-gated)
//!     ├── newton.rs         # WgpuClient impl
//!     ├── broyden.rs        # WgpuClient impl
//!     └── levenberg_marquardt.rs # WgpuClient impl
//! ```

mod cpu;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "wgpu")]
mod wgpu;

pub mod impl_generic;
pub mod traits;

// Re-export traits
pub use traits::{
    AndersonAlgorithms, AndersonOptions, Broyden1Algorithms, LevenbergMarquardtAlgorithms,
    NewtonSystemAlgorithms, PowellHybridAlgorithms,
};

// Re-export result type from impl_generic
pub use impl_generic::TensorRootResult;

/// Alias for TensorRootResult for convenience
pub type RootTensorResult<R> = TensorRootResult<R>;

/// Options for multivariate root finding.
#[derive(Debug, Clone)]
pub struct RootOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence (norm of F(x))
    pub tol: f64,
    /// Tolerance for step size
    pub x_tol: f64,
    /// Step size for finite difference Jacobian approximation
    pub eps: f64,
}

impl Default for RootOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-8,
            x_tol: 1e-8,
            eps: 1e-8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_options() {
        let opts = RootOptions::default();
        assert_eq!(opts.max_iter, 100);
        assert!((opts.tol - 1e-8).abs() < 1e-12);
    }
}
