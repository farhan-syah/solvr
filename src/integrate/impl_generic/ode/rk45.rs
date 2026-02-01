//! Dormand-Prince RK45 method using tensor operations.
//!
//! All computation stays on device using numr's TensorOps.
//! Step size control is fully device-resident - no scalar transfers during stepping.

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::{ODEMethod, ODEOptions};

use super::{
    ODEResultTensor, compute_acceptance, compute_error, compute_initial_step, compute_step_factor,
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

// Step size controller parameters
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 10.0;

/// Compute weighted sum of stages: sum(coeffs[i] * stages[i])
///
/// Uses tensor operations - data stays on device.
fn weighted_sum<R, C>(
    client: &C,
    stages: &[&Tensor<R>],
    coeffs: &[f64],
    h: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    debug_assert_eq!(stages.len(), coeffs.len());

    // h * coeffs[0] * stages[0]
    let h_c0 = client.mul_scalar(h, coeffs[0])?;
    let mut result = client.mul(&h_c0, stages[0])?;

    for i in 1..stages.len() {
        if coeffs[i] != 0.0 {
            let h_ci = client.mul_scalar(h, coeffs[i])?;
            let term = client.mul(&h_ci, stages[i])?;
            result = client.add(&result, &term)?;
        }
    }
    Ok(result)
}

/// Dormand-Prince RK45 method using tensor operations.
///
/// All computation stays on device. Step size control is fully device-resident.
/// Only control flow decisions (accept/reject) require minimal host-side scalar reads.
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
    F: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
{
    let [t_start, t_end] = t_span;
    let device = client.device();

    let min_step = options.min_step.unwrap_or(1e-14);
    let max_step = options.max_step.unwrap_or(t_end - t_start);

    // Initialize - all tensors stay on device
    let mut t = Tensor::<R>::from_slice(&[t_start], &[1], device);
    let mut y = y0.clone();
    let mut k1 = f(&t, &y).map_err(|e| IntegrateError::InvalidInput {
        context: format!("RHS function error: {}", e),
    })?;

    // Compute initial step size (device-resident)
    let mut h = match options.h0 {
        Some(h0) => Tensor::<R>::from_slice(&[h0], &[1], device),
        None => compute_initial_step(client, &f, &t, &y, &k1, 4, options.rtol, options.atol)
            .map_err(|e| IntegrateError::InvalidInput {
                context: format!("Initial step computation error: {}", e),
            })?,
    };

    // Clamp h to [min_step, max_step] on device
    let min_h = Tensor::<R>::from_slice(&[min_step], &[1], device);
    let max_h = Tensor::<R>::from_slice(&[max_step], &[1], device);
    h = client.minimum(&client.maximum(&h, &min_h)?, &max_h)?;

    // t_end tensor for comparison
    let t_end_tensor = Tensor::<R>::from_slice(&[t_end], &[1], device);

    // Storage for results
    let mut t_values = vec![t_start];
    let mut y_values = vec![y.clone()];
    let mut nfev = 1;
    let mut naccept = 0;
    let mut nreject = 0;

    // Main integration loop
    // Note: We transfer only t_val for loop condition and accept for control flow
    loop {
        let t_val: f64 = t.to_vec()[0];

        if t_val >= t_end {
            break;
        }

        if naccept + nreject >= options.max_steps {
            let (t_tensor, y_tensor) = build_result_tensors(client, &t_values, &y_values)?;
            return Ok(ODEResultTensor {
                t: t_tensor,
                y: y_tensor,
                success: false,
                message: Some(format!(
                    "Maximum steps ({}) exceeded at t = {:.6}",
                    options.max_steps, t_val
                )),
                nfev,
                naccept,
                nreject,
                method: ODEMethod::RK45,
            });
        }

        // Adjust step for end point: h = min(h, t_end - t)
        let remaining = client.sub(&t_end_tensor, &t)?;
        h = client.minimum(&h, &remaining)?;

        // ============================================================
        // RK45 stages - ALL computation stays on device
        // ============================================================

        // k2 = f(t + c2*h, y + h*a21*k1)
        let h_a21 = client.mul_scalar(&h, A21)?;
        let y_stage = client.add(&y, &client.mul(&h_a21, &k1)?)?;
        let t_stage = client.add(&t, &client.mul_scalar(&h, C2)?)?;
        let k2 = f(&t_stage, &y_stage).map_err(to_integrate_err)?;

        // k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
        let sum_k = weighted_sum(client, &[&k1, &k2], &[A31, A32], &h)?;
        let y_stage = client.add(&y, &sum_k)?;
        let t_stage = client.add(&t, &client.mul_scalar(&h, C3)?)?;
        let k3 = f(&t_stage, &y_stage).map_err(to_integrate_err)?;

        // k4 = f(t + c4*h, y + h*(a41*k1 + a42*k2 + a43*k3))
        let sum_k = weighted_sum(client, &[&k1, &k2, &k3], &[A41, A42, A43], &h)?;
        let y_stage = client.add(&y, &sum_k)?;
        let t_stage = client.add(&t, &client.mul_scalar(&h, C4)?)?;
        let k4 = f(&t_stage, &y_stage).map_err(to_integrate_err)?;

        // k5 = f(t + c5*h, y + h*(a51*k1 + a52*k2 + a53*k3 + a54*k4))
        let sum_k = weighted_sum(client, &[&k1, &k2, &k3, &k4], &[A51, A52, A53, A54], &h)?;
        let y_stage = client.add(&y, &sum_k)?;
        let t_stage = client.add(&t, &client.mul_scalar(&h, C5)?)?;
        let k5 = f(&t_stage, &y_stage).map_err(to_integrate_err)?;

        // k6 = f(t + h, y + h*(a61*k1 + a62*k2 + a63*k3 + a64*k4 + a65*k5))
        let sum_k = weighted_sum(
            client,
            &[&k1, &k2, &k3, &k4, &k5],
            &[A61, A62, A63, A64, A65],
            &h,
        )?;
        let y_stage = client.add(&y, &sum_k)?;
        let t_new = client.add(&t, &h)?;
        let k6 = f(&t_new, &y_stage).map_err(to_integrate_err)?;

        // 5th order solution: y5 = y + h*(b1*k1 + b3*k3 + b4*k4 + b5*k5 + b6*k6)
        let sum_b = weighted_sum(
            client,
            &[&k1, &k3, &k4, &k5, &k6],
            &[B1, B3, B4, B5, B6],
            &h,
        )?;
        let y5 = client.add(&y, &sum_b)?;

        // k7 (FSAL) = f(t + h, y5)
        let k7 = f(&t_new, &y5).map_err(to_integrate_err)?;
        nfev += 6;

        // Error estimate: y_err = h*(e1*k1 + e3*k3 + e4*k4 + e5*k5 + e6*k6 + e7*k7)
        let y_err = weighted_sum(
            client,
            &[&k1, &k3, &k4, &k5, &k6, &k7],
            &[E1, E3, E4, E5, E6, E7],
            &h,
        )?;

        // ============================================================
        // Device-resident step control
        // ============================================================

        // Compute error (stays on device as tensor)
        let error = compute_error(client, &y5, &y_err, &y, options.rtol, options.atol)
            .map_err(to_integrate_err)?;

        // Compute step factor (stays on device)
        let factor = compute_step_factor(client, &error, 4, SAFETY, MIN_FACTOR, MAX_FACTOR)
            .map_err(to_integrate_err)?;

        // Compute acceptance indicator (stays on device)
        let accept_tensor = compute_acceptance(client, &error).map_err(to_integrate_err)?;

        // Only transfer accept for control flow decision
        let accept_val: f64 = accept_tensor.to_vec()[0];
        let accept = accept_val > 0.5;

        // Compute new step size on device
        let h_new = client.mul(&h, &factor)?;

        // Clamp h_new on device
        let h_new = client.minimum(&client.maximum(&h_new, &min_h)?, &max_h)?;

        if accept {
            t = t_new;
            y = y5;
            k1 = k7; // FSAL property

            let new_t: f64 = t.to_vec()[0];
            t_values.push(new_t);
            y_values.push(y.clone());
            naccept += 1;
        } else {
            nreject += 1;
        }

        h = h_new;

        // Check minimum step (transfer h for comparison)
        let h_val: f64 = h.to_vec()[0];
        if h_val < min_step {
            return Err(IntegrateError::StepSizeTooSmall {
                step: h_val,
                t: t.to_vec()[0],
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

    let t_tensor = Tensor::<R>::from_slice(t_values, &[n_steps], client.device());

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
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0], &[2], &device);

        let opts = ODEOptions::with_tolerances(1e-6, 1e-8);

        let result = rk45_impl(
            &client,
            |_t, y| {
                // For harmonic oscillator: dy1/dt = y2, dy2/dt = -y1
                let y_data: Vec<f64> = y.to_vec();
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
                // 3t^2 - computed on device
                let t_sq = client.mul(t, t)?;
                client.mul_scalar(&t_sq, 3.0)
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
