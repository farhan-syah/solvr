//! Generic implementations of optimization algorithms.
//!
//! These implementations work across all Runtime backends by using
//! numr's tensor operations.

pub mod bfgs;
pub mod conjugate_gradient;
pub mod global;
pub mod helpers;
pub mod least_squares;
pub mod nelder_mead;
pub mod powell;
pub mod roots;
pub mod scalar;
pub mod utils;

// Re-export main types for convenience
pub use bfgs::bfgs_impl;
pub use conjugate_gradient::conjugate_gradient_impl;
pub use global::{TensorGlobalResult, simulated_annealing_impl, differential_evolution_impl};
pub use helpers::TensorMinimizeResult;
pub use least_squares::{TensorLeastSquaresResult, leastsq_impl, least_squares_impl};
pub use nelder_mead::nelder_mead_impl;
pub use powell::powell_impl;
pub use roots::{TensorRootResult, newton_system_impl, broyden1_impl, levenberg_marquardt_impl};
