//! Generic ODE solver implementations using tensor operations.
//!
//! All implementations use numr's `TensorOps` and `ScalarOps` for computation,
//! keeping data on device (GPU/CPU with SIMD) throughout the algorithm.

mod dop853;
mod rk23;
mod rk45;
mod step_control;

pub use dop853::dop853_impl;
pub use rk23::rk23_impl;
pub use rk45::rk45_impl;
pub use step_control::*;

use numr::error::Result;
use numr::runtime::Runtime;
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

/// Solve an initial value problem using tensor operations.
///
/// All computation stays on device. The RHS function `f` receives and returns tensors.
/// Time is passed as a scalar tensor (shape [1]) to enable device-resident computation.
///
/// # Arguments
///
/// * `client` - Runtime client for tensor operations
/// * `f` - Right-hand side function f(t, y) -> dy/dt, where t is a scalar tensor [1]
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
///     |_t, y| client.mul_scalar(y, -1.0),  // t is a tensor [1], y is a tensor [n]
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
    C: numr::ops::TensorOps<R> + numr::ops::ScalarOps<R> + numr::runtime::RuntimeClient<R>,
    F: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
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
        ODEMethod::DOP853 => dop853_impl(client, f, t_span, y0, options),
    }
}
