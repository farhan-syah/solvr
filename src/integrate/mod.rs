//! Numerical integration and ODE solvers for solvr.
//!
//! This module provides numerical methods for:
//! - **Quadrature**: Numerical integration of functions (definite integrals)
//! - **ODE Solvers**: Initial value problem solvers for ordinary differential equations
//!
//! # Quadrature Methods
//!
//! ## Basic Methods
//!
//! - [`trapezoid`] - Trapezoidal rule for uniformly spaced data
//! - [`cumulative_trapezoid`] - Cumulative trapezoidal integration
//! - [`simpson`] - Simpson's rule (higher accuracy than trapezoid)
//!
//! ## Gaussian Quadrature
//!
//! - [`fixed_quad`] - Fixed-order Gaussian quadrature
//! - [`GaussLegendreQuadrature`] - Gauss-Legendre quadrature with arbitrary points
//!
//! ## Adaptive Quadrature
//!
//! - [`quad`] - Adaptive Gauss-Kronrod quadrature (recommended for general use)
//! - [`romberg`] - Romberg integration (Richardson extrapolation)
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
//! # Examples
//!
//! ## Numerical Integration
//!
//! ```ignore
//! use solvr::integrate::{quad, QuadOptions};
//!
//! // Integrate sin(x) from 0 to pi
//! let result = quad(|x| x.sin(), 0.0, std::f64::consts::PI, &QuadOptions::default())?;
//! assert!((result.integral - 2.0).abs() < 1e-10);
//! ```
//!
//! ## Solving ODEs
//!
//! ```ignore
//! use solvr::integrate::{solve_ivp, ODEMethod, ODEOptions};
//!
//! // Solve dy/dt = -y, y(0) = 1
//! let result = solve_ivp(
//!     |t, y| vec![-y[0]],    // dy/dt = -y
//!     [0.0, 5.0],            // time span
//!     &[1.0],                // initial condition
//!     &ODEOptions::default(),
//! )?;
//! // y(5) ≈ e^(-5) ≈ 0.00674
//! ```

pub mod error;
pub mod ode;
pub mod quadrature;

// Re-export error types
pub use error::{IntegrateError, IntegrateResult};

// Re-export quadrature functions
pub use quadrature::{
    GaussLegendreQuadrature,
    QuadOptions,
    QuadResult,
    RombergOptions,
    // Basic quadrature
    cumulative_trapezoid,
    // Gaussian quadrature
    fixed_quad,
    // Adaptive quadrature
    quad,
    romberg,
    simpson,
    trapezoid,
};

// Re-export ODE types and functions
pub use ode::{ODEMethod, ODEOptions, ODEResult, ODESolution, StepSizeController, solve_ivp};
