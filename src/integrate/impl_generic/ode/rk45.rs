//! Dormand-Prince RK45 method using tensor operations.
//!
//! All computation stays on device using numr's TensorOps.

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::{ODEMethod, ODEOptions};

use super::{
    ODEResultTensor, StepSizeController, compute_error_tensor, compute_initial_step_tensor,
};

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

/// Compute weighted sum of stages: sum(coeffs[i] * stages[i])
///
/// Uses tensor operations - data stays on device.
fn weighted_sum<R, C>(
    client: &C,
    stages: &[&Tensor<R>],
    coeffs: &[f64],
    h: f64,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    debug_assert_eq!(stages.len(), coeffs.len());

    let mut result = client.mul_scalar(stages[0], h * coeffs[0])?;
    for i in 1..stages.len() {
        if coeffs[i] != 0.0 {
            let term = client.mul_scalar(stages[i], h * coeffs[i])?;
            result = client.add(&result, &term)?;
        }
    }
    Ok(result)
}

/// Dormand-Prince RK45 method using tensor operations.
///
/// All computation stays on device. No GPU→CPU→GPU roundtrips in the loop.
pub fn rk45_impl<R, C, F>(
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

    let controller = StepSizeController::default();
    let min_step = options.min_step.unwrap_or(1e-14);
    let max_step = options.max_step.unwrap_or(t_end - t_start);

    // Initialize - all tensors stay on device
    let mut t = t_start;
    let mut y = y0.clone();
    let mut k1 = f(t, &y).map_err(|e| IntegrateError::InvalidInput {
        context: format!("RHS function error: {}", e),
    })?;

    // Compute initial step size
    let mut h = match options.h0 {
        Some(h0) => h0,
        None => compute_initial_step_tensor(client, &f, t, &y, &k1, 4, options.rtol, options.atol)
            .map_err(|e| IntegrateError::InvalidInput {
                context: format!("Initial step computation error: {}", e),
            })?,
    };
    h = h.clamp(min_step, max_step);

    // Storage for results - we'll collect t and y values
    // and build tensors at the end to minimize allocations
    let mut t_values = vec![t];
    let mut y_values = vec![y.clone()];
    let mut nfev = 1;
    let mut naccept = 0;
    let mut nreject = 0;

    while t < t_end {
        if naccept + nreject >= options.max_steps {
            // Build result tensors from collected values
            let (t_tensor, y_tensor) = build_result_tensors(client, &t_values, &y_values)?;

            return Ok(ODEResultTensor {
                t: t_tensor,
                y: y_tensor,
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

        // ============================================================
        // RK45 stages - ALL computation stays on device
        // ============================================================

        // k2 = f(t + c2*h, y + h*a21*k1)
        let y_stage = client
            .add(&y, &client.mul_scalar(&k1, h * A21)?)
            .map_err(to_integrate_err)?;
        let k2 = f(t + C2 * h, &y_stage).map_err(to_integrate_err)?;

        // k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
        let sum_k = weighted_sum(client, &[&k1, &k2], &[A31, A32], h).map_err(to_integrate_err)?;
        let y_stage = client.add(&y, &sum_k).map_err(to_integrate_err)?;
        let k3 = f(t + C3 * h, &y_stage).map_err(to_integrate_err)?;

        // k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
        let sum_k = weighted_sum(client, &[&k1, &k2, &k3], &[A41, A42, A43], h)
            .map_err(to_integrate_err)?;
        let y_stage = client.add(&y, &sum_k).map_err(to_integrate_err)?;
        let k4 = f(t + C4 * h, &y_stage).map_err(to_integrate_err)?;

        // k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        let sum_k = weighted_sum(client, &[&k1, &k2, &k3, &k4], &[A51, A52, A53, A54], h)
            .map_err(to_integrate_err)?;
        let y_stage = client.add(&y, &sum_k).map_err(to_integrate_err)?;
        let k5 = f(t + C5 * h, &y_stage).map_err(to_integrate_err)?;

        // k6 = f(t + h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        let sum_k = weighted_sum(
            client,
            &[&k1, &k2, &k3, &k4, &k5],
            &[A61, A62, A63, A64, A65],
            h,
        )
        .map_err(to_integrate_err)?;
        let y_stage = client.add(&y, &sum_k).map_err(to_integrate_err)?;
        let k6 = f(t + h, &y_stage).map_err(to_integrate_err)?;

        // 5th order solution: y5 = y + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
        // Note: b2 = 0, so we skip k2
        let sum_b = weighted_sum(client, &[&k1, &k3, &k4, &k5, &k6], &[B1, B3, B4, B5, B6], h)
            .map_err(to_integrate_err)?;
        let y5 = client.add(&y, &sum_b).map_err(to_integrate_err)?;

        // k7 (FSAL) = f(t + h, y5)
        let k7 = f(t + h, &y5).map_err(to_integrate_err)?;
        nfev += 6;

        // Error estimate: y_err = h*(e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6 + e7*k7)
        let y_err = weighted_sum(
            client,
            &[&k1, &k3, &k4, &k5, &k6, &k7],
            &[E1, E3, E4, E5, E6, E7],
            h,
        )
        .map_err(to_integrate_err)?;

        let err = compute_error_tensor(client, &y5, &y_err, &y, options.rtol, options.atol)
            .map_err(to_integrate_err)?;

        let (h_new, accept) = controller.compute_step(h, err, 4);

        if accept {
            t += h;
            y = y5;
            k1 = k7; // FSAL property

            t_values.push(t);
            y_values.push(y.clone());
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

    // Build result tensors from collected values
    let (t_tensor, y_tensor) = build_result_tensors(client, &t_values, &y_values)?;

    Ok(ODEResultTensor {
        t: t_tensor,
        y: y_tensor,
        success: true,
        message: None,
        nfev,
        naccept,
        nreject,
        method: ODEMethod::RK45,
    })
}

/// Build result tensors from collected values.
///
/// Uses numr's `stack` operation - data stays on device, no CPU transfer.
fn build_result_tensors<R, C>(
    client: &C,
    t_values: &[f64],
    y_values: &[Tensor<R>],
) -> IntegrateResult<(Tensor<R>, Tensor<R>)>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let n_steps = t_values.len();

    // Build t tensor (scalar times are small, this is fine)
    let t_tensor = Tensor::<R>::from_slice(t_values, &[n_steps], client.device());

    // Stack y tensors on device - NO CPU transfer
    // Each y_values[i] is shape [n], stack gives [n_steps, n]
    let y_refs: Vec<&Tensor<R>> = y_values.iter().collect();
    let y_tensor = client
        .stack(&y_refs, 0)
        .map_err(|e| IntegrateError::InvalidInput {
            context: format!("Failed to stack y tensors: {}", e),
        })?;

    Ok((t_tensor, y_tensor))
}

/// Convert numr error to IntegrateError.
fn to_integrate_err(e: numr::error::Error) -> IntegrateError {
    IntegrateError::InvalidInput {
        context: format!("Tensor operation error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_rk45_exponential_decay() {
        let (device, client) = setup();

        // dy/dt = -y, y(0) = 1, solution: y(t) = exp(-t)
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);

        let result = rk45_impl(
            &client,
            |_t, y| client.mul_scalar(y, -1.0),
            [0.0, 5.0],
            &y0,
            &ODEOptions::default(),
        )
        .unwrap();

        assert!(result.success);
        assert_eq!(result.method, ODEMethod::RK45);

        // Get final value
        let y_val = result.y_final_vec();
        let exact = (-5.0_f64).exp();

        assert!(
            (y_val[0] - exact).abs() < 1e-4,
            "y_final = {}, exact = {}",
            y_val[0],
            exact
        );
    }

    #[test]
    fn test_rk45_harmonic_oscillator() {
        let (device, client) = setup();

        // y'' + y = 0 as system: y1' = y2, y2' = -y1
        // with y1(0) = 1, y2(0) = 0
        // solution: y1 = cos(t), y2 = -sin(t)
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0], &[2], &device);

        let opts = ODEOptions::with_tolerances(1e-6, 1e-8);

        let result = rk45_impl(
            &client,
            |_t, y| {
                let y_data: Vec<f64> = y.to_vec();
                // dy1/dt = y2, dy2/dt = -y1
                Ok(Tensor::<CpuRuntime>::from_slice(
                    &[y_data[1], -y_data[0]],
                    &[2],
                    &device,
                ))
            },
            [0.0, 2.0 * std::f64::consts::PI],
            &y0,
            &opts,
        )
        .unwrap();

        assert!(result.success);

        // After one period, should return to initial state
        let y_val = result.y_final_vec();

        assert!((y_val[0] - 1.0).abs() < 0.01, "y1 = {}", y_val[0]);
        assert!(y_val[1].abs() < 0.01, "y2 = {}", y_val[1]);
    }

    #[test]
    fn test_rk45_polynomial() {
        let (device, client) = setup();

        // dy/dt = 3t^2, y(0) = 0, solution: y(t) = t^3
        let y0 = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);

        let result = rk45_impl(
            &client,
            |t, _y| {
                Ok(Tensor::<CpuRuntime>::from_slice(
                    &[3.0 * t * t],
                    &[1],
                    &device,
                ))
            },
            [0.0, 2.0],
            &y0,
            &ODEOptions::default(),
        )
        .unwrap();

        assert!(result.success);

        let y_val = result.y_final_vec();

        assert!((y_val[0] - 8.0).abs() < 1e-6, "y_final = {}", y_val[0]);
    }
}
