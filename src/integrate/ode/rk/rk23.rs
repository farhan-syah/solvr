//! Bogacki-Shampine RK23 method.

#![allow(clippy::needless_range_loop)]

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::ode::types::{ODEMethod, ODEOptions, ODEResult};

use super::{StepSizeController, compute_error, compute_initial_step};

/// Bogacki-Shampine RK23 method.
///
/// A 2(3) adaptive Runge-Kutta method. Lower order and faster per step than RK45,
/// but typically requires more steps for the same accuracy.
pub fn rk23<F>(
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

    // Bogacki-Shampine coefficients
    const C2: f64 = 0.5;
    const C3: f64 = 0.75;

    const A21: f64 = 0.5;
    const A31: f64 = 0.0;
    const A32: f64 = 0.75;
    const A41: f64 = 2.0 / 9.0;
    const A42: f64 = 1.0 / 3.0;
    const A43: f64 = 4.0 / 9.0;

    // 3rd order weights
    const B1: f64 = 2.0 / 9.0;
    const B2: f64 = 1.0 / 3.0;
    const B3: f64 = 4.0 / 9.0;

    // Error weights (3rd - 2nd order)
    const E1: f64 = -5.0 / 72.0;
    const E2: f64 = 1.0 / 12.0;
    const E3: f64 = 1.0 / 9.0;
    const E4: f64 = -1.0 / 8.0;

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
        .unwrap_or_else(|| compute_initial_step(&f, t, &y, &k1, 2, options.rtol, options.atol));
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
                method: ODEMethod::RK23,
            });
        }

        // Adjust step for end point
        h = h.min(t_end - t);

        // RK23 stages
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

        // k4 and new y
        let mut y_new = vec![0.0; n];
        for i in 0..n {
            y_new[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
        }
        let k4 = f(t + h, &y_new);
        nfev += 3;

        // 3rd order solution
        let mut y3 = vec![0.0; n];
        for i in 0..n {
            y3[i] = y[i] + h * (B1 * k1[i] + B2 * k2[i] + B3 * k3[i]);
        }

        // Error estimate
        let mut y_err = vec![0.0; n];
        for i in 0..n {
            y_err[i] = h * (E1 * k1[i] + E2 * k2[i] + E3 * k3[i] + E4 * k4[i]);
        }

        let err = compute_error(&y3, &y_err, &y, options.rtol, options.atol);
        let (h_new, accept) = controller.compute_step(h, err, 2);

        if accept {
            t += h;
            y = y3;
            k1 = k4; // FSAL property

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
                context: "RK23".to_string(),
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
        method: ODEMethod::RK23,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rk23_exponential() {
        let options = ODEOptions::with_method(ODEMethod::RK23);

        let result = rk23(|_t, y| vec![-y[0]], [0.0, 2.0], &[1.0], &options).unwrap();

        assert!(result.success);
        assert_eq!(result.method, ODEMethod::RK23);

        let y_final = result.y.last().unwrap()[0];
        let exact = (-2.0_f64).exp();
        assert!((y_final - exact).abs() < 1e-3);
    }

    #[test]
    fn test_rk23_linear() {
        // dy/dt = 2, y(0) = 0, solution: y(t) = 2t
        let options = ODEOptions::with_method(ODEMethod::RK23);

        let result = rk23(|_t, _y| vec![2.0], [0.0, 5.0], &[0.0], &options).unwrap();

        assert!(result.success);

        let y_final = result.y.last().unwrap()[0];
        assert!((y_final - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_rk23_quadratic() {
        // dy/dt = 2t, y(0) = 0, solution: y(t) = t^2
        let options = ODEOptions::with_method(ODEMethod::RK23);

        let result = rk23(|t, _y| vec![2.0 * t], [0.0, 3.0], &[0.0], &options).unwrap();

        assert!(result.success);

        let y_final = result.y.last().unwrap()[0];
        assert!((y_final - 9.0).abs() < 1e-4);
    }
}
