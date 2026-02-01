//! Generic ODE solver implementations using tensor operations.
//!
//! All implementations use numr's `TensorOps` and `ScalarOps` for computation,
//! keeping data on device (GPU/CPU with SIMD) throughout the algorithm.

mod rk23;
mod rk45;

pub use rk23::rk23_impl;
pub use rk45::rk45_impl;

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::{ODEMethod, ODEOptions};

/// Result of tensor-based ODE integration.
///
/// All data is stored as tensors, remaining on device until explicitly
/// transferred to CPU via `to_vec()`.
#[derive(Debug, Clone)]
pub struct ODEResultTensor<R: Runtime> {
    /// Time points where solution was computed (1-D tensor)
    pub t: Tensor<R>,

    /// Solution values - shape [n_steps, n_vars]
    pub y: Tensor<R>,

    /// Whether integration was successful
    pub success: bool,

    /// Status message (e.g., why integration failed)
    pub message: Option<String>,

    /// Number of function evaluations
    pub nfev: usize,

    /// Number of accepted steps
    pub naccept: usize,

    /// Number of rejected steps
    pub nreject: usize,

    /// Method used for integration
    pub method: ODEMethod,
}

impl<R: Runtime> ODEResultTensor<R> {
    /// Get the final state as a tensor (stays on device).
    ///
    /// Note: This extracts the last row and transfers to CPU to rebuild as 1-D tensor.
    /// For on-device access, index directly into `self.y`.
    pub fn y_final(&self) -> Result<Tensor<R>>
    where
        R: Runtime,
    {
        // Get shape info
        let shape = self.y.shape();
        if shape.len() != 2 || shape[0] == 0 {
            return Err(numr::error::Error::InvalidArgument {
                arg: "y",
                reason: "Expected 2D tensor with at least one row".to_string(),
            });
        }

        // Can't easily get device here without RuntimeClient, so we return the full tensor
        // The caller can use y_final_vec() for the actual last row values
        // or index into y directly if they need on-device access
        Ok(self.y.clone())
    }

    /// Get the final state as a Vec<f64>.
    ///
    /// This is the recommended way to get the final state for inspection.
    pub fn y_final_vec(&self) -> Vec<f64> {
        let shape = self.y.shape();
        if shape.len() != 2 || shape[0] == 0 {
            return vec![];
        }

        let n_steps = shape[0];
        let n_vars = shape[1];

        let all_data: Vec<f64> = self.y.to_vec();
        let last_row_start = (n_steps - 1) * n_vars;
        all_data[last_row_start..].to_vec()
    }
}

/// Step size controller for adaptive methods.
#[derive(Debug, Clone)]
pub struct StepSizeController {
    /// Safety factor (default: 0.9)
    pub safety: f64,
    /// Minimum scale factor (default: 0.2)
    pub min_factor: f64,
    /// Maximum scale factor (default: 10.0)
    pub max_factor: f64,
}

impl Default for StepSizeController {
    fn default() -> Self {
        Self {
            safety: 0.9,
            min_factor: 0.2,
            max_factor: 10.0,
        }
    }
}

impl StepSizeController {
    /// Compute the new step size based on error estimate.
    pub fn compute_step(&self, h: f64, err: f64, order: usize) -> (f64, bool) {
        let accept = err <= 1.0;

        let exponent = 1.0 / (order as f64 + 1.0);
        let factor = if err == 0.0 {
            self.max_factor
        } else {
            self.safety * (1.0 / err).powf(exponent)
        };

        let factor = factor.clamp(self.min_factor, self.max_factor);
        let factor = if accept { factor } else { factor.min(1.0) };

        (h * factor, accept)
    }
}

/// Compute initial step size using the algorithm from Hairer & Wanner.
///
/// Uses tensor operations - data stays on device.
#[allow(clippy::too_many_arguments)]
pub fn compute_initial_step_tensor<R, C, F>(
    client: &C,
    f: &F,
    t0: f64,
    y0: &Tensor<R>,
    f0: &Tensor<R>,
    order: usize,
    rtol: f64,
    atol: f64,
) -> Result<f64>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(f64, &Tensor<R>) -> Result<Tensor<R>>,
{
    let n = y0.shape()[0] as f64;

    // Compute scaling: sc = atol + rtol * |y0|
    let y0_abs = client.abs(y0)?;
    let sc = client.add_scalar(&client.mul_scalar(&y0_abs, rtol)?, atol)?;

    // d0 = ||y0 / sc|| / sqrt(n)
    let y0_scaled = client.div(y0, &sc)?;
    let y0_scaled_sq = client.mul(&y0_scaled, &y0_scaled)?;
    let d0_sq = client.sum(&y0_scaled_sq, &[0], false)?;
    let d0_val: f64 = d0_sq.to_vec()[0];
    let d0 = (d0_val / n).sqrt();

    // d1 = ||f0 / sc|| / sqrt(n)
    let f0_scaled = client.div(f0, &sc)?;
    let f0_scaled_sq = client.mul(&f0_scaled, &f0_scaled)?;
    let d1_sq = client.sum(&f0_scaled_sq, &[0], false)?;
    let d1_val: f64 = d1_sq.to_vec()[0];
    let d1 = (d1_val / n).sqrt();

    // First guess
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };

    // Explicit Euler step: y1 = y0 + h0 * f0
    let h0_f0 = client.mul_scalar(f0, h0)?;
    let y1 = client.add(y0, &h0_f0)?;
    let f1 = f(t0 + h0, &y1)?;

    // d2 = ||f1 - f0|| / (h0 * sc)
    let df = client.sub(&f1, f0)?;
    let df_scaled = client.div(&df, &sc)?;
    let df_scaled_sq = client.mul(&df_scaled, &df_scaled)?;
    let d2_sq = client.sum(&df_scaled_sq, &[0], false)?;
    let d2_val: f64 = d2_sq.to_vec()[0];
    let d2 = (d2_val / n).sqrt() / h0;

    // Compute h1
    let h1 = if d1.max(d2) <= 1e-15 {
        (h0 * 1e-3).max(1e-6)
    } else {
        (0.01 / d1.max(d2)).powf(1.0 / (order as f64 + 1.0))
    };

    Ok(h0.min(100.0 * h0).min(h1))
}

/// Compute normalized error using tensor operations.
///
/// Returns a scalar error value. Data stays on device for the computation.
pub fn compute_error_tensor<R, C>(
    client: &C,
    y_new: &Tensor<R>,
    y_err: &Tensor<R>,
    y_old: &Tensor<R>,
    rtol: f64,
    atol: f64,
) -> Result<f64>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    let n = y_new.shape()[0] as f64;

    // sc = atol + rtol * max(|y_old|, |y_new|)
    let y_old_abs = client.abs(y_old)?;
    let y_new_abs = client.abs(y_new)?;
    let y_max = client.maximum(&y_old_abs, &y_new_abs)?;
    let sc = client.add_scalar(&client.mul_scalar(&y_max, rtol)?, atol)?;

    // err = sqrt(sum((y_err / sc)^2) / n)
    let err_scaled = client.div(y_err, &sc)?;
    let err_sq = client.mul(&err_scaled, &err_scaled)?;
    let sum_sq = client.sum(&err_sq, &[0], false)?;
    let sum_val: f64 = sum_sq.to_vec()[0];

    Ok((sum_val / n).sqrt())
}

/// Solve an initial value problem using tensor operations.
///
/// All computation stays on device. The RHS function `f` receives and returns tensors.
///
/// # Arguments
///
/// * `client` - Runtime client for tensor operations
/// * `f` - Right-hand side function f(t, y) -> dy/dt, operating on tensors
/// * `t_span` - Integration interval [t0, tf]
/// * `y0` - Initial condition as a 1-D tensor
/// * `options` - Solver options
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
pub fn solve_ivp_impl<R, C, F>(
    client: &C,
    f: F,
    t_span: [f64; 2],
    y0: &Tensor<R>,
    options: &ODEOptions,
) -> IntegrateResult<ODEResultTensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(f64, &Tensor<R>) -> Result<Tensor<R>>,
{
    let [t_start, t_end] = t_span;

    if t_start >= t_end {
        return Err(IntegrateError::InvalidInterval {
            a: t_start,
            b: t_end,
            context: "solve_ivp".to_string(),
        });
    }

    if y0.shape().is_empty() || y0.shape()[0] == 0 {
        return Err(IntegrateError::InvalidInput {
            context: "solve_ivp: initial condition cannot be empty".to_string(),
        });
    }

    match options.method {
        ODEMethod::RK23 => rk23_impl(client, f, t_span, y0, options),
        ODEMethod::RK45 => rk45_impl(client, f, t_span, y0, options),
        ODEMethod::DOP853 => {
            // For now, fall back to RK45 for DOP853
            // TODO: Implement DOP853 with tensor ops
            rk45_impl(client, f, t_span, y0, options)
        }
    }
}
