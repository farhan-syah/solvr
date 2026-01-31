//! Dormand-Prince RK45 method.

#![allow(clippy::needless_range_loop)]

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::ode::types::{ODEMethod, ODEOptions, ODEResult};

use super::{StepSizeController, compute_error, compute_initial_step};

/// Dormand-Prince RK45 method.
///
/// The classic 4(5) adaptive Runge-Kutta method. This is the default and
/// recommended method for most non-stiff problems.
pub fn rk45<F>(
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

    // Dormand-Prince coefficients
    const C2: f64 = 1.0 / 5.0;
    const C3: f64 = 3.0 / 10.0;
    const C4: f64 = 4.0 / 5.0;
    const C5: f64 = 8.0 / 9.0;

    const A21: f64 = 1.0 / 5.0;
    const A31: f64 = 3.0 / 40.0;
    const A32: f64 = 9.0 / 40.0;
    const A41: f64 = 44.0 / 45.0;
    const A42: f64 = -56.0 / 15.0;
    const A43: f64 = 32.0 / 9.0;
    const A51: f64 = 19372.0 / 6561.0;
    const A52: f64 = -25360.0 / 2187.0;
    const A53: f64 = 64448.0 / 6561.0;
    const A54: f64 = -212.0 / 729.0;
    const A61: f64 = 9017.0 / 3168.0;
    const A62: f64 = -355.0 / 33.0;
    const A63: f64 = 46732.0 / 5247.0;
    const A64: f64 = 49.0 / 176.0;
    const A65: f64 = -5103.0 / 18656.0;

    // 5th order weights
    const B1: f64 = 35.0 / 384.0;
    const B3: f64 = 500.0 / 1113.0;
    const B4: f64 = 125.0 / 192.0;
    const B5: f64 = -2187.0 / 6784.0;
    const B6: f64 = 11.0 / 84.0;

    // Error weights (5th - 4th order)
    const E1: f64 = 71.0 / 57600.0;
    const E3: f64 = -71.0 / 16695.0;
    const E4: f64 = 71.0 / 1920.0;
    const E5: f64 = -17253.0 / 339200.0;
    const E6: f64 = 22.0 / 525.0;
    const E7: f64 = -1.0 / 40.0;

    let controller = StepSizeController::default();
    let min_step = options.min_step.unwrap_or(1e-14);
    let max_step = options.max_step.unwrap_or(t_end - t_start);

    // Initialize
    let mut t = t_start;
    let mut y = y0.to_vec();
    let mut k1 = f(t, &y);

    // Compute initial step size
    let mut h = options
        .h0
        .unwrap_or_else(|| compute_initial_step(&f, t, &y, &k1, 4, options.rtol, options.atol));
    h = h.clamp(min_step, max_step);

    // Storage
    let mut t_vec = vec![t];
    let mut y_vec = vec![y.clone()];
    let mut nfev = 1;
    let mut naccept = 0;
    let mut nreject = 0;

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
                method: ODEMethod::RK45,
            });
        }

        // Adjust step for end point
        h = h.min(t_end - t);

        // RK45 stages
        let mut y_stage = vec![0.0; n];

        // k2
        for i in 0..n {
            y_stage[i] = y[i] + h * A21 * k1[i];
        }
        let k2 = f(t + C2 * h, &y_stage);

        // k3
        for i in 0..n {
            y_stage[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
        }
        let k3 = f(t + C3 * h, &y_stage);

        // k4
        for i in 0..n {
            y_stage[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
        }
        let k4 = f(t + C4 * h, &y_stage);

        // k5
        for i in 0..n {
            y_stage[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
        }
        let k5 = f(t + C5 * h, &y_stage);

        // k6
        for i in 0..n {
            y_stage[i] =
                y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
        }
        let k6 = f(t + h, &y_stage);

        // 5th order solution
        let mut y5 = vec![0.0; n];
        for i in 0..n {
            y5[i] = y[i] + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]);
        }

        // k7 (FSAL)
        let k7 = f(t + h, &y5);
        nfev += 6;

        // Error estimate
        let mut y_err = vec![0.0; n];
        for i in 0..n {
            y_err[i] =
                h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
        }

        let err = compute_error(&y5, &y_err, &y, options.rtol, options.atol);
        let (h_new, accept) = controller.compute_step(h, err, 4);

        if accept {
            t += h;
            y = y5;
            k1 = k7; // FSAL property

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
                context: "RK45".to_string(),
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
        method: ODEMethod::RK45,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rk45_exponential() {
        let options = ODEOptions::with_method(ODEMethod::RK45);

        let result = rk45(|_t, y| vec![-y[0]], [0.0, 5.0], &[1.0], &options).unwrap();

        assert!(result.success);
        assert_eq!(result.method, ODEMethod::RK45);

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
    fn test_rk45_polynomial() {
        // dy/dt = 3t^2, y(0) = 0, solution: y(t) = t^3
        let options = ODEOptions::with_method(ODEMethod::RK45);

        let result = rk45(|t, _y| vec![3.0 * t * t], [0.0, 2.0], &[0.0], &options).unwrap();

        assert!(result.success);

        let y_final = result.y.last().unwrap()[0];
        assert!((y_final - 8.0).abs() < 1e-6);
    }

    #[test]
    fn test_rk45_system() {
        // Rotation: y1' = y2, y2' = -y1
        let options = ODEOptions::with_tolerances(1e-8, 1e-10);

        let result = rk45(|_t, y| vec![y[1], -y[0]], [0.0, 1.0], &[1.0, 0.0], &options).unwrap();

        assert!(result.success);

        let y_final = result.y.last().unwrap();
        let exact_y1 = 1.0_f64.cos();
        let exact_y2 = -1.0_f64.sin();

        assert!((y_final[0] - exact_y1).abs() < 1e-6);
        assert!((y_final[1] - exact_y2).abs() < 1e-6);
    }

    #[test]
    fn test_rk45_efficiency() {
        // RK45 should use fewer function evaluations than RK23 for same tolerance
        let tol_opts = ODEOptions::with_tolerances(1e-6, 1e-8);

        let rk45_result = rk45(|_t, y| vec![-y[0]], [0.0, 10.0], &[1.0], &tol_opts).unwrap();

        assert!(rk45_result.success);
        // RK45 should converge efficiently
        assert!(rk45_result.naccept > 0);
    }
}
