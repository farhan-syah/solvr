//! DOP853: Dormand-Prince 8(5,3) method.
//!
//! A 12-stage explicit Runge-Kutta method of order 8 with embedded
//! 5th and 3rd order error estimators. Suitable for high-accuracy requirements.
//!
//! Reference: Hairer, Nørsett, Wanner "Solving Ordinary Differential Equations I"

mod coefficients;

use coefficients::*;

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::ode::rk::{StepSizeController, compute_initial_step};
use crate::integrate::ode::types::{ODEMethod, ODEOptions, ODEResult};

/// Compute error norm using DOP853's dual error estimator.
/// Uses both 5th and 3rd order estimates for stability.
fn compute_error_norm(
    k: &[Vec<f64>],
    h: f64,
    y_old: &[f64],
    y_new: &[f64],
    rtol: f64,
    atol: f64,
) -> f64 {
    let n = y_old.len();
    let mut err5_norm_sq = 0.0;
    let mut err3_norm_sq = 0.0;

    for i in 0..n {
        let scale = atol + rtol * y_old[i].abs().max(y_new[i].abs());

        // Compute err5 and err3 for this component
        let mut err5 = 0.0;
        let mut err3 = 0.0;
        for j in 0..N_STAGES {
            err5 += E5[j] * k[j][i];
            err3 += E3[j] * k[j][i];
        }
        // k[12] contribution (FSAL, same as k_new)
        err5 += E5[N_STAGES] * k[N_STAGES][i];
        err3 += E3[N_STAGES] * k[N_STAGES][i];

        err5 /= scale;
        err3 /= scale;

        err5_norm_sq += err5 * err5;
        err3_norm_sq += err3 * err3;
    }

    if err5_norm_sq == 0.0 && err3_norm_sq == 0.0 {
        return 0.0;
    }

    let denom = err5_norm_sq + 0.01 * err3_norm_sq;
    h.abs() * err5_norm_sq / (denom * n as f64).sqrt()
}

/// Dormand-Prince 8(5,3) method.
///
/// High-order explicit Runge-Kutta method for demanding accuracy requirements.
/// Uses 12 stages per step with dual 5th/3rd order error estimators.
pub fn dop853<F>(
    f: F,
    t_span: [f64; 2],
    y0: &[f64],
    options: &ODEOptions,
) -> IntegrateResult<ODEResult>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let [t_start, t_end] = t_span;
    let n = y0.len();

    let controller = StepSizeController {
        safety: 0.9,
        min_factor: 0.2,
        max_factor: 10.0,
    };

    let min_step = options.min_step.unwrap_or(1e-14);
    let max_step = options.max_step.unwrap_or(t_end - t_start);

    // Initialize
    let mut t = t_start;
    let mut y = y0.to_vec();
    let mut k: Vec<Vec<f64>> = vec![vec![0.0; n]; N_STAGES + 1];
    k[0] = f(t, &y);

    // Compute initial step size
    let mut h = options
        .h0
        .unwrap_or_else(|| compute_initial_step(&f, t, &y, &k[0], 7, options.rtol, options.atol));
    h = h.clamp(min_step, max_step);

    // Storage
    let mut t_vec = vec![t];
    let mut y_vec = vec![y.clone()];
    let mut nfev = 1;
    let mut naccept = 0;
    let mut nreject = 0;

    let mut y_stage = vec![0.0; n];

    while t < t_end {
        if naccept + nreject >= options.max_steps {
            return Ok(ODEResult {
                t: t_vec,
                y: y_vec,
                success: false,
                message: Some(format!(
                    "Maximum steps ({}) exceeded at t = {:.6}",
                    options.max_steps, t
                )),
                nfev,
                naccept,
                nreject,
                method: ODEMethod::DOP853,
            });
        }

        h = h.min(t_end - t);

        // Stage 2
        for i in 0..n {
            y_stage[i] = y[i] + h * A2[0] * k[0][i];
        }
        k[1] = f(t + C[1] * h, &y_stage);

        // Stage 3
        for i in 0..n {
            y_stage[i] = y[i] + h * (A3[0] * k[0][i] + A3[1] * k[1][i]);
        }
        k[2] = f(t + C[2] * h, &y_stage);

        // Stage 4
        for i in 0..n {
            y_stage[i] = y[i] + h * (A4_0 * k[0][i] + A4_2 * k[2][i]);
        }
        k[3] = f(t + C[3] * h, &y_stage);

        // Stage 5
        for i in 0..n {
            y_stage[i] = y[i] + h * (A5_0 * k[0][i] + A5_2 * k[2][i] + A5_3 * k[3][i]);
        }
        k[4] = f(t + C[4] * h, &y_stage);

        // Stage 6
        for i in 0..n {
            y_stage[i] = y[i] + h * (A6_0 * k[0][i] + A6_3 * k[3][i] + A6_4 * k[4][i]);
        }
        k[5] = f(t + C[5] * h, &y_stage);

        // Stage 7
        for i in 0..n {
            y_stage[i] =
                y[i] + h * (A7_0 * k[0][i] + A7_3 * k[3][i] + A7_4 * k[4][i] + A7_5 * k[5][i]);
        }
        k[6] = f(t + C[6] * h, &y_stage);

        // Stage 8
        for i in 0..n {
            y_stage[i] = y[i]
                + h * (A8_0 * k[0][i]
                    + A8_3 * k[3][i]
                    + A8_4 * k[4][i]
                    + A8_5 * k[5][i]
                    + A8_6 * k[6][i]);
        }
        k[7] = f(t + C[7] * h, &y_stage);

        // Stage 9
        for i in 0..n {
            y_stage[i] = y[i]
                + h * (A9_0 * k[0][i]
                    + A9_3 * k[3][i]
                    + A9_4 * k[4][i]
                    + A9_5 * k[5][i]
                    + A9_6 * k[6][i]
                    + A9_7 * k[7][i]);
        }
        k[8] = f(t + C[8] * h, &y_stage);

        // Stage 10
        for i in 0..n {
            y_stage[i] = y[i]
                + h * (A10_0 * k[0][i]
                    + A10_3 * k[3][i]
                    + A10_4 * k[4][i]
                    + A10_5 * k[5][i]
                    + A10_6 * k[6][i]
                    + A10_7 * k[7][i]
                    + A10_8 * k[8][i]);
        }
        k[9] = f(t + C[9] * h, &y_stage);

        // Stage 11
        for i in 0..n {
            y_stage[i] = y[i]
                + h * (A11_0 * k[0][i]
                    + A11_3 * k[3][i]
                    + A11_4 * k[4][i]
                    + A11_5 * k[5][i]
                    + A11_6 * k[6][i]
                    + A11_7 * k[7][i]
                    + A11_8 * k[8][i]
                    + A11_9 * k[9][i]);
        }
        k[10] = f(t + C[10] * h, &y_stage);

        // Stage 12
        for i in 0..n {
            y_stage[i] = y[i]
                + h * (A12_0 * k[0][i]
                    + A12_3 * k[3][i]
                    + A12_4 * k[4][i]
                    + A12_5 * k[5][i]
                    + A12_6 * k[6][i]
                    + A12_7 * k[7][i]
                    + A12_8 * k[8][i]
                    + A12_9 * k[9][i]
                    + A12_10 * k[10][i]);
        }
        k[11] = f(t + C[11] * h, &y_stage);

        // Compute 8th order solution
        let mut y_new = vec![0.0; n];
        for i in 0..n {
            y_new[i] = y[i]
                + h * (B[0] * k[0][i]
                    + B[5] * k[5][i]
                    + B[6] * k[6][i]
                    + B[7] * k[7][i]
                    + B[8] * k[8][i]
                    + B[9] * k[9][i]
                    + B[10] * k[10][i]
                    + B[11] * k[11][i]);
        }

        // FSAL: k[12] = f(t + h, y_new)
        k[N_STAGES] = f(t + h, &y_new);
        nfev += 12;

        // Compute error norm using dual estimator
        let err = compute_error_norm(&k, h, &y, &y_new, options.rtol, options.atol);
        let (h_new, accept) = controller.compute_step(h, err, 7);

        if accept {
            t += h;
            y = y_new;
            k[0] = k[N_STAGES].clone(); // FSAL

            t_vec.push(t);
            y_vec.push(y.clone());
            naccept += 1;
        } else {
            nreject += 1;
        }

        h = h_new.clamp(min_step, max_step);

        if h < min_step {
            return Err(IntegrateError::StepSizeTooSmall {
                step: h,
                t,
                context: "DOP853".to_string(),
            });
        }
    }

    Ok(ODEResult {
        t: t_vec,
        y: y_vec,
        success: true,
        message: None,
        nfev,
        naccept,
        nreject,
        method: ODEMethod::DOP853,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dop853_exponential() {
        // dy/dt = -y, y(0) = 1, solution: y(t) = exp(-t)
        let opts = ODEOptions::with_method(ODEMethod::DOP853);
        let result = dop853(|_t, y| vec![-y[0]], [0.0, 5.0], &[1.0], &opts).unwrap();

        assert!(result.success, "Solver failed: {:?}", result.message);
        assert_eq!(result.method, ODEMethod::DOP853);

        let y_final = result.y.last().unwrap()[0];
        let exact = (-5.0_f64).exp();
        assert!(
            (y_final - exact).abs() < 1e-6,
            "y_final = {}, exact = {}, error = {}",
            y_final,
            exact,
            (y_final - exact).abs()
        );
    }

    #[test]
    fn test_dop853_harmonic_oscillator() {
        // y'' + y = 0: y1' = y2, y2' = -y1
        let opts = ODEOptions::with_tolerances(1e-8, 1e-10).method(ODEMethod::DOP853);
        let result = dop853(
            |_t, y| vec![y[1], -y[0]],
            [0.0, 2.0 * std::f64::consts::PI],
            &[1.0, 0.0],
            &opts,
        )
        .unwrap();

        assert!(result.success, "Solver failed: {:?}", result.message);

        let y_final = result.y.last().unwrap();
        assert!(
            (y_final[0] - 1.0).abs() < 1e-6,
            "y1 error: {}",
            (y_final[0] - 1.0).abs()
        );
        assert!(y_final[1].abs() < 1e-6, "y2 error: {}", y_final[1].abs());
    }

    #[test]
    fn test_dop853_polynomial() {
        // dy/dt = 3t^2, y(0) = 0, solution: y(t) = t^3
        let opts = ODEOptions::with_method(ODEMethod::DOP853);
        let result = dop853(|t, _y| vec![3.0 * t * t], [0.0, 2.0], &[0.0], &opts).unwrap();

        assert!(result.success);

        let y_final = result.y.last().unwrap()[0];
        let exact = 8.0;
        assert!(
            (y_final - exact).abs() < 1e-8,
            "error = {}",
            (y_final - exact).abs()
        );
    }

    #[test]
    fn test_dop853_linear_system() {
        // Rotation matrix
        let opts = ODEOptions::with_tolerances(1e-8, 1e-10);
        let result = dop853(|_t, s| vec![s[1], -s[0]], [0.0, 1.0], &[1.0, 0.0], &opts).unwrap();

        assert!(result.success);

        let y_final = result.y.last().unwrap();
        let exact_x = 1.0_f64.cos();
        let exact_y = -1.0_f64.sin();

        assert!((y_final[0] - exact_x).abs() < 1e-7);
        assert!((y_final[1] - exact_y).abs() < 1e-7);
    }
}
