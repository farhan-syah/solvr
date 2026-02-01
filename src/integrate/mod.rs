//! Numerical integration and ODE solvers for solvr.
//!
//! This module provides Runtime-first numerical methods for:
//! - **Quadrature**: Numerical integration of functions (definite integrals)
//! - **ODE Solvers**: Initial value problem solvers for ordinary differential equations
//!
//! # Architecture
//!
//! All algorithms implement the [`IntegrationAlgorithms`] trait and use
//! `Tensor<R>` for data, enabling GPU acceleration and batch operations.
//!
//! # Quadrature Methods
//!
//! - [`IntegrationAlgorithms::trapezoid`] - Trapezoidal rule with tensor support
//! - [`IntegrationAlgorithms::simpson`] - Simpson's rule with tensor support
//! - [`IntegrationAlgorithms::fixed_quad`] - Gaussian quadrature with tensor functions
//! - [`IntegrationAlgorithms::quad`] - Adaptive Gauss-Kronrod quadrature
//! - [`IntegrationAlgorithms::romberg`] - Romberg integration via Richardson extrapolation
//!
//! # ODE Solvers
//!
//! ## Unified Interface
//!
//! - [`solve_ivp`] - Main entry point for solving initial value problems
//!
//! ## Available Methods
//!
//! - **RK23**: Bogacki-Shampine 2(3) - Low accuracy, fast
//! - **RK45**: Dormand-Prince 4(5) - General purpose (default)
//! - **DOP853**: Dormand-Prince 8(5,3) - High accuracy
//!
//! # Example
//!
//! ```ignore
//! use solvr::integrate::IntegrationAlgorithms;
//! use numr::runtime::cpu::{CpuClient, CpuDevice};
//!
//! let device = CpuDevice::new();
//! let client = CpuClient::new(device.clone());
//!
//! // Tensor-based integration
//! let x = Tensor::from_slice(&[0.0, 0.5, 1.0], &[3], &device);
//! let y = Tensor::from_slice(&[0.0, 0.25, 1.0], &[3], &device);  // y = x^2
//! let result = client.trapezoid(&y, &x)?;
//! ```

mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
pub mod error;
pub mod impl_generic;
pub mod ode;
#[cfg(feature = "wgpu")]
mod wgpu;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

// Re-export error types
pub use error::{IntegrateError, IntegrateResult};

// Re-export ODE types
pub use ode::{ODEMethod, ODEOptions};

// Re-export tensor-based ODE types and functions
pub use impl_generic::ode::{ODEResultTensor, StepSizeController, solve_ivp_impl};

/// Options for adaptive quadrature.
#[derive(Debug, Clone)]
pub struct QuadOptions {
    /// Relative tolerance (default: 1e-8)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-8)
    pub atol: f64,
    /// Maximum number of subdivisions (default: 50)
    pub limit: usize,
}

impl Default for QuadOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-8,
            limit: 50,
        }
    }
}

/// Result of adaptive quadrature.
#[derive(Debug, Clone)]
pub struct QuadResult<R: Runtime> {
    /// Computed integral value (0-D tensor)
    pub integral: Tensor<R>,
    /// Estimated absolute error
    pub error: f64,
    /// Number of function evaluations
    pub neval: usize,
    /// Whether integration converged
    pub converged: bool,
}

/// Options for Romberg integration.
#[derive(Debug, Clone)]
pub struct RombergOptions {
    /// Relative tolerance (default: 1e-8)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-8)
    pub atol: f64,
    /// Maximum number of extrapolation levels (default: 20)
    pub max_levels: usize,
}

impl Default for RombergOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-8,
            max_levels: 20,
        }
    }
}

/// Trait for integration algorithms that work across all Runtime backends.
///
/// This trait provides a unified interface for:
/// - Trapezoidal integration
/// - Simpson's rule
/// - Gaussian quadrature
/// - Adaptive quadrature
/// - Romberg integration
///
/// All methods work with `Tensor<R>` for GPU acceleration and batch operations.
///
/// # Example
///
/// ```ignore
/// use solvr::integrate::IntegrationAlgorithms;
/// use numr::runtime::cpu::{CpuClient, CpuDevice};
///
/// let device = CpuDevice::new();
/// let client = CpuClient::new(device.clone());
///
/// // Integrate y = x^2 from 0 to 1
/// let x = Tensor::from_slice(&[0.0, 0.25, 0.5, 0.75, 1.0], &[5], &device);
/// let y = Tensor::from_slice(&[0.0, 0.0625, 0.25, 0.5625, 1.0], &[5], &device);
/// let result = client.trapezoid(&y, &x)?;
/// ```
pub trait IntegrationAlgorithms<R: Runtime> {
    /// Trapezoidal rule integration.
    ///
    /// Computes ∫y dx using the composite trapezoidal rule.
    ///
    /// # Arguments
    /// * `y` - Function values (1D or 2D for batch)
    /// * `x` - Sample points (1D)
    ///
    /// # Returns
    /// * 0-D tensor for 1D input
    /// * 1-D tensor for 2D input (one value per row)
    fn trapezoid(&self, y: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Trapezoidal rule with uniform spacing.
    ///
    /// # Arguments
    /// * `y` - Function values
    /// * `dx` - Uniform spacing between points
    fn trapezoid_uniform(&self, y: &Tensor<R>, dx: f64) -> Result<Tensor<R>>;

    /// Cumulative trapezoidal integration.
    ///
    /// Returns running integral values.
    ///
    /// # Arguments
    /// * `y` - Function values
    /// * `x` - Sample points (optional, uses dx if None)
    /// * `dx` - Uniform spacing (used if x is None)
    fn cumulative_trapezoid(
        &self,
        y: &Tensor<R>,
        x: Option<&Tensor<R>>,
        dx: f64,
    ) -> Result<Tensor<R>>;

    /// Simpson's rule integration.
    ///
    /// Uses Simpson's 1/3 rule for higher accuracy than trapezoidal.
    ///
    /// # Arguments
    /// * `y` - Function values
    /// * `x` - Sample points (optional, uses dx if None)
    /// * `dx` - Uniform spacing (used if x is None)
    fn simpson(&self, y: &Tensor<R>, x: Option<&Tensor<R>>, dx: f64) -> Result<Tensor<R>>;

    /// Fixed-order Gaussian quadrature.
    ///
    /// Integrates a tensor-valued function from a to b using
    /// n-point Gauss-Legendre quadrature.
    ///
    /// # Arguments
    /// * `f` - Function that takes tensor of evaluation points
    /// * `a` - Lower bound
    /// * `b` - Upper bound
    /// * `n` - Number of quadrature points
    fn fixed_quad<F>(&self, f: F, a: f64, b: f64, n: usize) -> Result<Tensor<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;

    /// Adaptive Gauss-Kronrod quadrature.
    ///
    /// Uses the G7-K15 rule (7-point Gauss, 15-point Kronrod) with adaptive
    /// interval subdivision to achieve the requested tolerance.
    ///
    /// # Arguments
    /// * `f` - Function that takes tensor of evaluation points and returns tensor of values
    /// * `a` - Lower bound
    /// * `b` - Upper bound
    /// * `options` - Quadrature options (tolerances, max subdivisions)
    ///
    /// # Returns
    /// A [`QuadResult`] containing the integral, error estimate, and diagnostics.
    fn quad<F>(&self, f: F, a: f64, b: f64, options: &QuadOptions) -> Result<QuadResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;

    /// Romberg integration using Richardson extrapolation.
    ///
    /// Applies Richardson extrapolation to the trapezoidal rule to achieve
    /// high accuracy for smooth functions.
    ///
    /// # Arguments
    /// * `f` - Function that takes tensor of evaluation points and returns tensor of values
    /// * `a` - Lower bound
    /// * `b` - Upper bound
    /// * `options` - Integration options (tolerances, max levels)
    ///
    /// # Returns
    /// A [`QuadResult`] containing the integral, error estimate, and diagnostics.
    fn romberg<F>(&self, f: F, a: f64, b: f64, options: &RombergOptions) -> Result<QuadResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<Tensor<R>>;

    /// Solve an initial value problem for ODEs using tensor operations.
    ///
    /// Solves the system dy/dt = f(t, y) with initial condition y(t0) = y0.
    /// All computation stays on device - no GPU→CPU→GPU roundtrips.
    ///
    /// # Arguments
    /// * `f` - Right-hand side function f(t, y) -> dy/dt, operating on tensors
    /// * `t_span` - Integration interval [t0, tf]
    /// * `y0` - Initial condition as a 1-D tensor
    /// * `options` - Solver options (method, tolerances, step bounds)
    ///
    /// # Returns
    /// An [`ODEResultTensor`] with solution trajectory stored as tensors.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use solvr::integrate::{IntegrationAlgorithms, ODEOptions};
    /// use numr::runtime::cpu::{CpuClient, CpuDevice};
    ///
    /// let device = CpuDevice::new();
    /// let client = CpuClient::new(device.clone());
    ///
    /// // Solve dy/dt = -y, y(0) = 1
    /// let y0 = Tensor::from_slice(&[1.0], &[1], &device);
    /// let result = client.solve_ivp(
    ///     |_t, y| client.mul_scalar(y, -1.0),
    ///     [0.0, 5.0],
    ///     &y0,
    ///     &ODEOptions::default(),
    /// )?;
    /// ```
    fn solve_ivp<F>(
        &self,
        f: F,
        t_span: [f64; 2],
        y0: &Tensor<R>,
        options: &ODEOptions,
    ) -> error::IntegrateResult<ODEResultTensor<R>>
    where
        F: Fn(f64, &Tensor<R>) -> Result<Tensor<R>>;
}
