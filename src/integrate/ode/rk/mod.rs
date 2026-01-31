//! Runge-Kutta ODE solvers.
//!
//! Implements RK23, RK45 (Dormand-Prince), and DOP853 methods.

mod rk23;
mod rk45;

pub use rk23::rk23;
pub use rk45::rk45;

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::ode::types::{ODEMethod, ODEOptions, ODEResult};

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
    ///
    /// # Arguments
    /// * `h` - Current step size
    /// * `err` - Normalized error (should be <= 1 for step acceptance)
    /// * `order` - Order of the error estimator
    pub fn compute_step(&self, h: f64, err: f64, order: usize) -> (f64, bool) {
        let accept = err <= 1.0;

        // Compute scale factor
        let exponent = 1.0 / (order as f64 + 1.0);
        let factor = if err == 0.0 {
            self.max_factor
        } else {
            self.safety * (1.0 / err).powf(exponent)
        };

        // Clamp factor
        let factor = factor.clamp(self.min_factor, self.max_factor);

        // Don't increase step size after rejection
        let factor = if accept { factor } else { factor.min(1.0) };

        (h * factor, accept)
    }
}

/// Solve an initial value problem.
///
/// Main entry point for ODE solving. Solves the system:
///   dy/dt = f(t, y)
///   y(t0) = y0
///
/// # Arguments
///
/// * `f` - Right-hand side function f(t, y) -> dy/dt
/// * `t_span` - Integration interval [t0, tf]
/// * `y0` - Initial condition
/// * `options` - Solver options
///
/// # Returns
///
/// An [`ODEResult`] containing the solution trajectory and diagnostics.
///
/// # Example
///
/// ```
/// use solvr::integrate::{solve_ivp, ODEOptions};
///
/// // Solve dy/dt = -y, y(0) = 1
/// // Exact solution: y(t) = exp(-t)
/// let result = solve_ivp(
///     |_t, y| vec![-y[0]],
///     [0.0, 5.0],
///     &[1.0],
///     &ODEOptions::default(),
/// ).unwrap();
///
/// let y_final = result.y.last().unwrap()[0];
/// let exact = (-5.0_f64).exp();
/// assert!((y_final - exact).abs() < 1e-4);
/// ```
pub fn solve_ivp<F>(
    f: F,
    t_span: [f64; 2],
    y0: &[f64],
    options: &ODEOptions,
) -> IntegrateResult<ODEResult>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let [t_start, t_end] = t_span;

    if t_start >= t_end {
        return Err(IntegrateError::InvalidInterval {
            a: t_start,
            b: t_end,
            context: "solve_ivp".to_string(),
        });
    }

    if y0.is_empty() {
        return Err(IntegrateError::InvalidInput {
            context: "solve_ivp: initial condition cannot be empty".to_string(),
        });
    }

    match options.method {
        ODEMethod::RK23 => rk23(f, t_span, y0, options),
        ODEMethod::RK45 => rk45(f, t_span, y0, options),
        ODEMethod::DOP853 => super::dop853::dop853(f, t_span, y0, options),
    }
}

/// Compute initial step size using the algorithm from Hairer & Wanner.
pub fn compute_initial_step<F>(
    f: &F,
    t0: f64,
    y0: &[f64],
    f0: &[f64],
    order: usize,
    rtol: f64,
    atol: f64,
) -> f64
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let n = y0.len();

    // Compute scaling
    let mut sc = vec![0.0; n];
    for i in 0..n {
        sc[i] = atol + rtol * y0[i].abs();
    }

    // Compute d0 = ||y0 / sc||
    let d0: f64 = y0
        .iter()
        .zip(&sc)
        .map(|(y, s)| (y / s).powi(2))
        .sum::<f64>()
        .sqrt()
        / (n as f64).sqrt();

    // Compute d1 = ||f0 / sc||
    let d1: f64 = f0
        .iter()
        .zip(&sc)
        .map(|(f, s)| (f / s).powi(2))
        .sum::<f64>()
        .sqrt()
        / (n as f64).sqrt();

    // First guess
    let h0 = if d0 < 1e-5 || d1 < 1e-5 {
        1e-6
    } else {
        0.01 * d0 / d1
    };

    // Explicit Euler step to estimate second derivative
    let y1: Vec<f64> = y0.iter().zip(f0).map(|(y, f)| y + h0 * f).collect();
    let f1 = f(t0 + h0, &y1);

    // Compute d2 = ||f1 - f0|| / (h0 * ||sc||)
    let d2: f64 = f1
        .iter()
        .zip(f0)
        .zip(&sc)
        .map(|((f1, f0), s)| ((f1 - f0) / s).powi(2))
        .sum::<f64>()
        .sqrt()
        / (h0 * (n as f64).sqrt());

    // Compute h1
    let h1 = if d1.max(d2) <= 1e-15 {
        (h0 * 1e-3).max(1e-6)
    } else {
        (0.01 / d1.max(d2)).powf(1.0 / (order as f64 + 1.0))
    };

    h0.min(100.0 * h0).min(h1)
}

/// Compute normalized error.
pub fn compute_error(y_new: &[f64], y_err: &[f64], y_old: &[f64], rtol: f64, atol: f64) -> f64 {
    let n = y_new.len();
    let mut err_sum = 0.0;

    for i in 0..n {
        let sc = atol + rtol * y_old[i].abs().max(y_new[i].abs());
        err_sum += (y_err[i] / sc).powi(2);
    }

    (err_sum / n as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_decay() {
        // dy/dt = -y, y(0) = 1, solution: y(t) = exp(-t)
        let result = solve_ivp(
            |_t, y| vec![-y[0]],
            [0.0, 5.0],
            &[1.0],
            &ODEOptions::default(),
        )
        .unwrap();

        assert!(result.success);

        // Check final value
        let y_final = result.y.last().unwrap()[0];
        let exact = (-5.0_f64).exp();
        assert!(
            (y_final - exact).abs() < 1e-4,
            "y_final = {}, exact = {}",
            y_final,
            exact
        );
    }

    #[test]
    fn test_harmonic_oscillator() {
        // y'' + y = 0, written as system:
        // y1' = y2
        // y2' = -y1
        // with y1(0) = 1, y2(0) = 0
        // solution: y1 = cos(t), y2 = -sin(t)

        // Use tighter tolerances for this long integration
        let opts = ODEOptions::with_tolerances(1e-6, 1e-8);
        let result = solve_ivp(
            |_t, y| vec![y[1], -y[0]],
            [0.0, 2.0 * std::f64::consts::PI],
            &[1.0, 0.0],
            &opts,
        )
        .unwrap();

        assert!(result.success);

        // After one period, should return to initial state
        let y_final = result.y.last().unwrap();
        assert!((y_final[0] - 1.0).abs() < 0.01, "y1 = {}", y_final[0]);
        assert!(y_final[1].abs() < 0.01, "y2 = {}", y_final[1]);
    }

    #[test]
    fn test_van_der_pol() {
        // Van der Pol oscillator (mu = 0.5, mildly stiff)
        // y1' = y2
        // y2' = mu * (1 - y1^2) * y2 - y1

        let mu = 0.5;
        let result = solve_ivp(
            move |_t, y| vec![y[1], mu * (1.0 - y[0].powi(2)) * y[1] - y[0]],
            [0.0, 20.0],
            &[2.0, 0.0],
            &ODEOptions::default(),
        )
        .unwrap();

        assert!(result.success);
        assert!(result.nfev > 0);
    }

    #[test]
    fn test_step_size_controller() {
        let controller = StepSizeController::default();

        // Accept step with small error
        let (h_new, accept) = controller.compute_step(0.1, 0.1, 4);
        assert!(accept);
        assert!(h_new > 0.1); // Should increase step

        // Reject step with large error
        let (h_new, accept) = controller.compute_step(0.1, 10.0, 4);
        assert!(!accept);
        assert!(h_new < 0.1); // Should decrease step
    }

    #[test]
    fn test_invalid_input() {
        // Invalid interval
        let result = solve_ivp(
            |_t, y| vec![-y[0]],
            [5.0, 0.0],
            &[1.0],
            &ODEOptions::default(),
        );
        assert!(result.is_err());

        // Empty initial condition
        let result = solve_ivp(|_t, _y| vec![], [0.0, 1.0], &[], &ODEOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_linear_system() {
        // d/dt [y1, y2] = [[0, 1], [-1, 0]] * [y1, y2]
        // Rotation matrix, solution is [cos(t), -sin(t)] for y0 = [1, 0]

        let result = solve_ivp(
            |_t, y| vec![y[1], -y[0]],
            [0.0, 1.0],
            &[1.0, 0.0],
            &ODEOptions::with_tolerances(1e-8, 1e-10),
        )
        .unwrap();

        assert!(result.success);

        let y_final = result.y.last().unwrap();
        let exact_y1 = 1.0_f64.cos();
        let exact_y2 = -1.0_f64.sin();

        assert!((y_final[0] - exact_y1).abs() < 1e-6);
        assert!((y_final[1] - exact_y2).abs() < 1e-6);
    }
}
