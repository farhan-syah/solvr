//! Ordinary differential equation (ODE) solvers.
//!
//! This module provides methods for solving initial value problems (IVPs)
//! of the form dy/dt = f(t, y), y(t0) = y0.
//!
//! # Available Methods
//!
//! | Method | Order | Type | Best For |
//! |--------|-------|------|----------|
//! | RK23 | 2(3) | Explicit | Low accuracy, fast |
//! | RK45 | 4(5) | Explicit | General purpose (default) |
//! | DOP853 | 8(5,3) | Explicit | High accuracy |
//!
//! # Usage
//!
//! Use [`solve_ivp`] as the main entry point. It provides a unified interface
//! to all ODE solving methods.
//!
//! ```ignore
//! use solvr::integrate::{solve_ivp, ODEMethod, ODEOptions};
//!
//! // Solve dy/dt = -y, y(0) = 1
//! let result = solve_ivp(
//!     |_t, y| vec![-y[0]],  // RHS function
//!     [0.0, 5.0],           // time span
//!     &[1.0],               // initial condition
//!     &ODEOptions::default(),
//! )?;
//!
//! // y(5) ≈ exp(-5) ≈ 0.00674
//! assert!((result.y.last().unwrap()[0] - (-5.0_f64).exp()).abs() < 1e-5);
//! ```

pub mod dop853;
mod rk;
mod types;

pub use rk::{StepSizeController, compute_error, compute_initial_step, solve_ivp};
pub use types::{ODEMethod, ODEOptions, ODEResult, ODESolution};
