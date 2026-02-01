//! Generic implementations of optimization algorithms.
//!
//! These implementations work across all Runtime backends by using
//! numr's tensor operations.

pub mod bfgs;
pub mod conjugate_gradient;
pub mod helpers;
// NOTE: least_squares module has been restructured and moved to src/optimize/least_squares/impl_generic/
// It is no longer declared here.
pub mod nelder_mead;
pub mod powell;
pub mod scalar;
pub mod utils;

// Re-export main types for convenience
pub use bfgs::bfgs_impl;
pub use conjugate_gradient::conjugate_gradient_impl;
pub use helpers::TensorMinimizeResult;
pub use nelder_mead::nelder_mead_impl;
pub use powell::powell_impl;

// NOTE: least_squares has been restructured. The implementations are now in:
// - crate::optimize::least_squares::impl_generic for the generic implementations
// - crate::optimize::least_squares::cpu/cuda/wgpu for the backend-specific implementations
// Import via: use crate::optimize::least_squares::impl_generic::{leastsq_impl, least_squares_impl};
pub mod least_squares {
    //! Re-export least_squares implementations from new location for backward compatibility
    pub use crate::optimize::least_squares::impl_generic::{
        TensorLeastSquaresResult, least_squares_impl, leastsq_impl,
    };
}

// NOTE: roots has been restructured. The implementations are now in:
// - crate::optimize::roots::impl_generic for the generic implementations
// - crate::optimize::roots::cpu/cuda/wgpu for the backend-specific implementations
// Import via: use crate::optimize::roots::impl_generic::{newton_system_impl, broyden1_impl, levenberg_marquardt_impl};
pub mod roots {
    //! Re-export roots implementations from new location for backward compatibility
    pub use crate::optimize::roots::impl_generic::{
        TensorRootResult, broyden1_impl, levenberg_marquardt_impl, newton_system_impl,
    };
}

// NOTE: global has been restructured. The implementations are now in:
// - crate::optimize::global::impl_generic for the generic implementations
// - crate::optimize::global::traits for trait definitions
// - crate::optimize::global::cpu/cuda/wgpu for the backend-specific implementations
// Import the traits directly from crate::optimize::global
