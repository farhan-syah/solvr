//! Bogacki-Shampine RK23 method using tensor operations.
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

// Bogacki-Shampine coefficients
const C2: f64 = 0.5;
const C3: f64 = 0.75;

const A21: f64 = 0.5;
#[allow(dead_code)] // Part of Butcher tableau, kept for documentation
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

/// Compute weighted sum of stages: sum(coeffs[i] * stages[i])
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

/// Bogacki-Shampine RK23 method using tensor operations.
///
/// Lower order than RK45, faster per step but requires more steps.
/// All computation stays on device.
pub fn rk23_impl<R, C, F>(
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

    // Initialize
    let mut t = t_start;
    let mut y = y0.clone();
    let mut k1 = f(t, &y).map_err(|e| IntegrateError::InvalidInput {
        context: format!("RHS function error: {}", e),
    })?;

    // Compute initial step size
    let mut h = match options.h0 {
        Some(h0) => h0,
        None => compute_initial_step_tensor(client, &f, t, &y, &k1, 2, options.rtol, options.atol)
            .map_err(|e| IntegrateError::InvalidInput {
                context: format!("Initial step computation error: {}", e),
            })?,
    };
    h = h.clamp(min_step, max_step);

    // Storage
    let mut t_values = vec![t];
    let mut y_values = vec![y.clone()];
    let mut nfev = 1;
    let mut naccept = 0;
    let mut nreject = 0;

    while t < t_end {
        if naccept + nreject >= options.max_steps {
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
                method: ODEMethod::RK23,
            });
        }

        h = h.min(t_end - t);

        // ============================================================
        // RK23 stages - ALL computation stays on device
        // ============================================================

        // k2 = f(t + c2*h, y + h*a21*k1)
        let y_stage = client
            .add(&y, &client.mul_scalar(&k1, h * A21)?)
            .map_err(to_integrate_err)?;
        let k2 = f(t + C2 * h, &y_stage).map_err(to_integrate_err)?;

        // k3 = f(t + c3*h, y + h*(a31*k1 + a32*k2))
        // Note: A31 = 0, so we only need a32*k2
        let y_stage = client
            .add(&y, &client.mul_scalar(&k2, h * A32)?)
            .map_err(to_integrate_err)?;
        let k3 = f(t + C3 * h, &y_stage).map_err(to_integrate_err)?;

        // y_new = y + h*(a41*k1 + a42*k2 + a43*k3)
        let sum_a = weighted_sum(client, &[&k1, &k2, &k3], &[A41, A42, A43], h)
            .map_err(to_integrate_err)?;
        let y_new = client.add(&y, &sum_a).map_err(to_integrate_err)?;

        // k4 (FSAL) = f(t + h, y_new)
        let k4 = f(t + h, &y_new).map_err(to_integrate_err)?;
        nfev += 3;

        // 3rd order solution: y3 = y + h*(b1*k1 + b2*k2 + b3*k3)
        let sum_b =
            weighted_sum(client, &[&k1, &k2, &k3], &[B1, B2, B3], h).map_err(to_integrate_err)?;
        let y3 = client.add(&y, &sum_b).map_err(to_integrate_err)?;

        // Error estimate
        let y_err = weighted_sum(client, &[&k1, &k2, &k3, &k4], &[E1, E2, E3, E4], h)
            .map_err(to_integrate_err)?;

        let err = compute_error_tensor(client, &y3, &y_err, &y, options.rtol, options.atol)
            .map_err(to_integrate_err)?;

        let (h_new, accept) = controller.compute_step(h, err, 2);

        if accept {
            t += h;
            y = y3;
            k1 = k4; // FSAL

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
                context: "RK23".to_string(),
            });
        }
    }

    let (t_tensor, y_tensor) = build_result_tensors(client, &t_values, &y_values)?;

    Ok(ODEResultTensor {
        t: t_tensor,
        y: y_tensor,
        success: true,
        message: None,
        nfev,
        naccept,
        nreject,
        method: ODEMethod::RK23,
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
    let y_refs: Vec<&Tensor<R>> = y_values.iter().collect();
    let y_tensor = client
        .stack(&y_refs, 0)
        .map_err(|e| IntegrateError::InvalidInput {
            context: format!("Failed to stack y tensors: {}", e),
        })?;

    Ok((t_tensor, y_tensor))
}

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
    fn test_rk23_exponential() {
        let (device, client) = setup();

        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);

        let result = rk23_impl(
            &client,
            |_t, y| client.mul_scalar(y, -1.0),
            [0.0, 2.0],
            &y0,
            &ODEOptions::with_method(ODEMethod::RK23),
        )
        .unwrap();

        assert!(result.success);
        assert_eq!(result.method, ODEMethod::RK23);

        let y_val = result.y_final_vec();
        let exact = (-2.0_f64).exp();

        assert!((y_val[0] - exact).abs() < 1e-3);
    }

    #[test]
    fn test_rk23_linear() {
        let (device, client) = setup();

        // dy/dt = 2, y(0) = 0, solution: y(t) = 2t
        let y0 = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device);

        let result = rk23_impl(
            &client,
            |_t, _y| Ok(Tensor::<CpuRuntime>::from_slice(&[2.0], &[1], &device)),
            [0.0, 5.0],
            &y0,
            &ODEOptions::with_method(ODEMethod::RK23),
        )
        .unwrap();

        assert!(result.success);

        let y_val = result.y_final_vec();

        assert!((y_val[0] - 10.0).abs() < 1e-6);
    }
}
