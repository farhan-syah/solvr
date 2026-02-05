//! BDF-based DAE (Differential-Algebraic Equation) solver.
//!
//! Solves systems of the form F(t, y, y') = 0 using variable-order BDF methods
//! with automatic Jacobian computation via autograd.
//!
//! # Unique Capabilities
//!
//! - **Automatic Jacobians**: Both ∂F/∂y and ∂F/∂y' computed via forward-mode AD
//! - **Consistent IC**: Automatic refinement of initial conditions
//! - **GPU-compatible**: All computation via numr tensor ops
//!
//! # Example
//!
//! ```ignore
//! use numr::autograd::dual_ops::{dual_sub, dual_mul_scalar};
//!
//! // DAE: y' = z, 0 = y - sin(t)
//! // Formulated as F(t, y, y') = [y' - z, y - sin(t)]
//! let f = |t: &DualTensor<R>, y: &DualTensor<R>, yp: &DualTensor<R>, c: &C| {
//!     // Build residual using dual ops for autograd
//!     // ...
//! };
//!
//! let result = dae_impl(&client, f, [0.0, 10.0], &y0, &yp0, &ode_opts, &dae_opts)?;
//! ```

use numr::autograd::DualTensor;
use numr::error::Result;
use numr::ops::{LinalgOps, ScalarOps, TensorOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::ode::{DAEOptions, DAEResultTensor, DAEVariableType, ODEOptions};

use super::dae_ic::compute_consistent_ic;
use super::dae_jacobian::{compute_dae_jacobian, eval_dae_primal};
use super::jacobian::compute_norm_scalar;

#[cfg(feature = "sparse")]
use super::sparse_utils::{dense_to_csr_full, solve_with_gmres};
#[cfg(feature = "sparse")]
use numr::algorithm::iterative::IterativeSolvers;
#[cfg(feature = "sparse")]
use numr::sparse::SparseOps;

// BDF coefficients (same as bdf.rs)
const BDF_ALPHA: [[f64; 6]; 5] = [
    [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
    [3.0 / 2.0, -2.0, 1.0 / 2.0, 0.0, 0.0, 0.0],
    [11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0, 0.0, 0.0],
    [25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 1.0 / 4.0, 0.0],
    [137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 5.0 / 4.0, -1.0 / 5.0],
];

const BDF_BETA: [f64; 5] = [1.0, 2.0 / 3.0, 6.0 / 11.0, 12.0 / 25.0, 60.0 / 137.0];
const BDF_ERROR_COEFF: [f64; 5] = [1.0 / 2.0, 2.0 / 9.0, 3.0 / 22.0, 12.0 / 125.0, 10.0 / 137.0];

const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.2;
const MAX_FACTOR: f64 = 5.0;

/// DAE solver implementation using BDF methods.
///
/// Solves F(t, y, y') = 0 with automatic Jacobian computation.
#[cfg(not(feature = "sparse"))]
pub fn dae_impl<R, C, F>(
    client: &C,
    f: F,
    t_span: [f64; 2],
    y0: &Tensor<R>,
    yp0: &Tensor<R>,
    options: &ODEOptions,
    dae_options: &DAEOptions<R>,
) -> IntegrateResult<DAEResultTensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinalgOps<R> + UtilityOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    dae_impl_inner(client, f, t_span, y0, yp0, options, dae_options)
}

/// DAE solver implementation with sparse support.
#[cfg(feature = "sparse")]
pub fn dae_impl<R, C, F>(
    client: &C,
    f: F,
    t_span: [f64; 2],
    y0: &Tensor<R>,
    yp0: &Tensor<R>,
    options: &ODEOptions,
    dae_options: &DAEOptions<R>,
) -> IntegrateResult<DAEResultTensor<R>>
where
    R: Runtime,
    C: TensorOps<R>
        + ScalarOps<R>
        + LinalgOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>
        + IterativeSolvers<R>
        + SparseOps<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    dae_impl_inner(client, f, t_span, y0, yp0, options, dae_options)
}

/// Inner implementation of DAE solver.
fn dae_impl_inner<R, C, F>(
    client: &C,
    f: F,
    t_span: [f64; 2],
    y0: &Tensor<R>,
    yp0: &Tensor<R>,
    options: &ODEOptions,
    dae_options: &DAEOptions<R>,
) -> IntegrateResult<DAEResultTensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinalgOps<R> + UtilityOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    let [t_start, t_end] = t_span;
    let device = client.device();
    let n = y0.shape()[0];

    // Validate inputs
    if t_start >= t_end {
        return Err(IntegrateError::InvalidInterval {
            a: t_start,
            b: t_end,
            context: "solve_dae".to_string(),
        });
    }

    if y0.shape() != yp0.shape() {
        return Err(IntegrateError::InvalidInput {
            context: format!(
                "y0 and yp0 must have same shape: {:?} vs {:?}",
                y0.shape(),
                yp0.shape()
            ),
        });
    }

    // Validate variable types if provided
    if dae_options
        .variable_types
        .as_ref()
        .is_some_and(|types| types.len() != n)
    {
        return Err(IntegrateError::InvalidInput {
            context: format!(
                "variable_types length {} doesn't match state dimension {}",
                dae_options
                    .variable_types
                    .as_ref()
                    .expect("variable_types validated above")
                    .len(),
                n
            ),
        });
    }

    let min_step = options.min_step.unwrap_or(1e-14);
    let max_step = options.max_step.unwrap_or(t_end - t_start);
    let max_order = dae_options.max_order.clamp(1, 5);

    // Compute consistent initial conditions
    let (mut y, mut yp, n_ic_iter) =
        compute_consistent_ic(client, &f, t_start, y0, yp0, dae_options)?;

    // Initialize state
    let mut t_val = t_start;

    // History buffers
    let mut y_history: Vec<Tensor<R>> = vec![y.clone()];
    let mut yp_history: Vec<Tensor<R>> = vec![yp.clone()];

    // Initial step size estimation
    let yp_norm = compute_norm_scalar(client, &yp, 2.0).unwrap_or(1.0);
    let y_norm = compute_norm_scalar(client, &y, 2.0).unwrap_or(1.0);
    let mut h = options
        .h0
        .unwrap_or_else(|| 0.01 * (y_norm / yp_norm.max(1e-10)).min(max_step).max(min_step));
    h = h.clamp(min_step, max_step);

    let mut order = 1;

    // Results storage
    let mut t_values = vec![t_val];
    let mut y_values = vec![y.clone()];
    let mut yp_values = if dae_options.return_yp {
        vec![yp.clone()]
    } else {
        vec![]
    };

    let mut nfev = 1; // IC computation counts
    let mut njac = 0;
    let mut naccept = 0;
    let mut nreject = 0;

    // Jacobian caching
    let mut jacobian: Option<Tensor<R>> = None;
    let mut steps_since_jacobian = 0;
    let jacobian_update_interval = 5;

    // Main integration loop
    while t_val < t_end {
        if naccept + nreject >= options.max_steps {
            return build_dae_result(
                client,
                &t_values,
                &y_values,
                &yp_values,
                false,
                Some(format!(
                    "Maximum steps ({}) exceeded at t = {:.6}",
                    options.max_steps, t_val
                )),
                nfev,
                njac,
                n_ic_iter,
                naccept,
                nreject,
                dae_options.return_yp,
            );
        }

        h = h.min(t_end - t_val);

        // Compute predictor for y using derivative information
        let y_pred = compute_predictor_with_yp(client, &y_history, &yp_history, h)?;
        let yp_pred = compute_yp_from_bdf(client, &y_pred, &y_history, order, h)?;

        // Update Jacobian if needed
        if jacobian.is_none() || steps_since_jacobian >= jacobian_update_interval {
            let t_new = Tensor::<R>::from_slice(&[t_val + h], &[1], device);
            let order_idx = (order - 1).min(4);
            let alpha0 = BDF_ALPHA[order_idx][0];
            let beta = BDF_BETA[order_idx];
            let coeff = alpha0 / (h * beta);

            jacobian = Some(
                compute_dae_jacobian(client, &f, &t_new, &y_pred, &yp_pred, coeff)
                    .map_err(to_integrate_err)?,
            );
            steps_since_jacobian = 0;
            njac += 1;
            nfev += 2 * n; // Forward-mode AD for y and yp Jacobians
        }

        // Newton iteration
        let (y_new, yp_new, converged, newton_iters) = dae_newton_iteration(
            client,
            &f,
            t_val + h,
            &y_pred,
            &y_history,
            order,
            h,
            jacobian.as_ref().expect("jacobian computed above"),
            dae_options,
        )?;
        nfev += newton_iters;

        if !converged {
            h *= 0.5;
            nreject += 1;
            jacobian = None;

            if h < min_step {
                return Err(IntegrateError::StepSizeTooSmall {
                    step: h,
                    t: t_val,
                    context: "DAE Newton iteration failed to converge".to_string(),
                });
            }
            continue;
        }

        // Error estimation
        let error_tensor = estimate_error(client, &y_new, &y_pred, order)?;

        // Optionally exclude algebraic variables from error
        let error_val = if dae_options.exclude_algebraic_from_error {
            compute_error_with_exclusion(
                client,
                &y_new,
                &error_tensor,
                &y,
                options.rtol,
                options.atol,
                &dae_options.variable_types,
            )
            .map_err(to_integrate_err)?
        } else {
            compute_error(
                client,
                &y_new,
                &error_tensor,
                &y,
                options.rtol,
                options.atol,
            )
            .map_err(to_integrate_err)?
        };

        if error_val <= 1.0 {
            // Accept step
            t_val += h;
            y = y_new;
            yp = yp_new;

            y_history.insert(0, y.clone());
            yp_history.insert(0, yp.clone());

            let history_len = max_order + 1;
            if y_history.len() > history_len {
                y_history.truncate(history_len);
                yp_history.truncate(history_len);
            }

            t_values.push(t_val);
            y_values.push(y.clone());
            if dae_options.return_yp {
                yp_values.push(yp.clone());
            }

            naccept += 1;
            steps_since_jacobian += 1;

            order = adjust_order(order, error_val, max_order, y_history.len());
        } else {
            nreject += 1;
            jacobian = None;
        }

        // Step size control
        let factor = compute_step_factor(error_val, order);
        h = (h * factor).clamp(min_step, max_step);
    }

    build_dae_result(
        client,
        &t_values,
        &y_values,
        &yp_values,
        true,
        None,
        nfev,
        njac,
        n_ic_iter,
        naccept,
        nreject,
        dae_options.return_yp,
    )
}

/// Compute predictor using polynomial extrapolation from history.
///
/// Uses derivative information for better prediction when starting.
fn compute_predictor_with_yp<R, C>(
    client: &C,
    y_history: &[Tensor<R>],
    yp_history: &[Tensor<R>],
    h: f64,
) -> IntegrateResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    if y_history.is_empty() || yp_history.is_empty() {
        return Err(IntegrateError::InvalidInput {
            context: "Empty history".to_string(),
        });
    }

    let y_n = &y_history[0];
    let yp_n = &yp_history[0];

    // First-order Taylor predictor: y_{n+1} ≈ y_n + h * yp_n
    // This is much better than just copying y_n
    let h_yp = client.mul_scalar(yp_n, h).map_err(to_integrate_err)?;
    client.add(y_n, &h_yp).map_err(to_integrate_err)
}

/// Compute y' from BDF formula: y'_{n+1} = (α₀·y_{n+1} + Σⱼ αⱼ·y_{n+1-j}) / (h·β)
fn compute_yp_from_bdf<R, C>(
    client: &C,
    y_new: &Tensor<R>,
    y_history: &[Tensor<R>],
    order: usize,
    h: f64,
) -> IntegrateResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    let order_idx = (order - 1).min(4);
    let alpha = &BDF_ALPHA[order_idx];
    let beta = BDF_BETA[order_idx];

    // yp = (α₀·y_{n+1} + Σⱼ αⱼ·y_{n+1-j}) / (h·β)
    let mut numerator = client
        .mul_scalar(y_new, alpha[0])
        .map_err(to_integrate_err)?;

    for i in 1..=order {
        if i <= y_history.len() {
            let term = client
                .mul_scalar(&y_history[i - 1], alpha[i])
                .map_err(to_integrate_err)?;
            numerator = client.add(&numerator, &term).map_err(to_integrate_err)?;
        }
    }

    client
        .mul_scalar(&numerator, 1.0 / (h * beta))
        .map_err(to_integrate_err)
}

/// Newton iteration for DAE step.
#[allow(clippy::too_many_arguments)]
fn dae_newton_iteration<R, C, F>(
    client: &C,
    f: &F,
    t_new: f64,
    y_pred: &Tensor<R>,
    y_history: &[Tensor<R>],
    order: usize,
    h: f64,
    jacobian: &Tensor<R>,
    dae_options: &DAEOptions<R>,
) -> IntegrateResult<(Tensor<R>, Tensor<R>, bool, usize)>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinalgOps<R> + UtilityOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &DualTensor<R>, &DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    let device = client.device();
    let n = y_pred.shape()[0];

    // The Jacobian passed in is already J = ∂F/∂y + coeff * ∂F/∂y' where coeff = α₀/(h·β)
    // For Newton iteration, we solve: J * Δy = -F(t, y, y')
    let iteration_matrix = jacobian;

    let mut y_iter = y_pred.clone();
    let t_tensor = Tensor::<R>::from_slice(&[t_new], &[1], device);
    let mut nfev = 0;
    let mut prev_res_norm = f64::MAX;

    for iter in 0..dae_options.max_newton_iter {
        // Compute y' from BDF formula
        let yp_iter = compute_yp_from_bdf(client, &y_iter, y_history, order, h)?;

        // Evaluate residual F(t, y, y')
        let residual =
            eval_dae_primal(client, f, &t_tensor, &y_iter, &yp_iter).map_err(to_integrate_err)?;
        nfev += 1;

        let res_norm = compute_norm_scalar(client, &residual, 2.0).map_err(to_integrate_err)?;
        let y_norm = compute_norm_scalar(client, &y_iter, 2.0).map_err(to_integrate_err)?;

        // Check convergence
        let tol = dae_options.newton_tol * (1.0 + y_norm);
        if res_norm < tol {
            return Ok((y_iter, yp_iter, true, nfev));
        }

        // Check if making progress - if residual increased, may need smaller steps
        if iter > 0 && res_norm > prev_res_norm * 2.0 {
            // Residual blowing up, return current iterate as not converged
            return Ok((y_iter, yp_iter, false, nfev));
        }
        prev_res_norm = res_norm;

        // Solve iteration_matrix * delta = -residual
        let neg_res = client
            .mul_scalar(&residual, -1.0)
            .map_err(to_integrate_err)?;
        let neg_res_col = neg_res.reshape(&[n, 1]).map_err(to_integrate_err)?;

        let delta_col = match solve_dae_linear(client, iteration_matrix, &neg_res_col, dae_options)
        {
            Ok(d) => d,
            Err(_) => {
                // Matrix solve failed (singular), return not converged
                return Ok((y_iter, yp_iter, false, nfev));
            }
        };
        let delta = delta_col.reshape(&[n]).map_err(to_integrate_err)?;

        // Update y
        y_iter = client.add(&y_iter, &delta).map_err(to_integrate_err)?;
    }

    // Return current iterate even if not converged
    let yp_final = compute_yp_from_bdf(client, &y_iter, y_history, order, h)?;
    Ok((y_iter, yp_final, false, nfev))
}

/// Solve linear system for DAE Newton step.
#[cfg(feature = "sparse")]
fn solve_dae_linear<R, C>(
    client: &C,
    m_dense: &Tensor<R>,
    b: &Tensor<R>,
    dae_options: &DAEOptions<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: LinalgOps<R> + RuntimeClient<R> + IterativeSolvers<R> + SparseOps<R>,
{
    if !dae_options.sparse_jacobian.enabled {
        return client.solve(m_dense, b);
    }

    let m_sparse = dense_to_csr_full(client, m_dense)?;
    solve_with_gmres(client, &m_sparse, b, &dae_options.sparse_jacobian, "DAE")
}

#[cfg(not(feature = "sparse"))]
fn solve_dae_linear<R, C>(
    client: &C,
    m_dense: &Tensor<R>,
    b: &Tensor<R>,
    _dae_options: &DAEOptions<R>,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: LinalgOps<R> + RuntimeClient<R>,
{
    client.solve(m_dense, b)
}

/// Estimate local truncation error.
fn estimate_error<R, C>(
    client: &C,
    y_new: &Tensor<R>,
    y_pred: &Tensor<R>,
    order: usize,
) -> IntegrateResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    let order_idx = (order - 1).min(4);
    let error_coeff = BDF_ERROR_COEFF[order_idx];
    let diff = client.sub(y_new, y_pred).map_err(to_integrate_err)?;
    client
        .mul_scalar(&diff, error_coeff)
        .map_err(to_integrate_err)
}

/// Compute normalized error (for all variables).
fn compute_error<R, C>(
    client: &C,
    y_new: &Tensor<R>,
    error: &Tensor<R>,
    y_old: &Tensor<R>,
    rtol: f64,
    atol: f64,
) -> Result<f64>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R>,
{
    let y_abs = client.abs(y_new)?;
    let y_old_abs = client.abs(y_old)?;
    let max_abs = client.maximum(&y_abs, &y_old_abs)?;
    let scale = client.add_scalar(&client.mul_scalar(&max_abs, rtol)?, atol)?;
    let normalized = client.div(error, &scale)?;
    let sq = client.mul(&normalized, &normalized)?;
    let mean_sq = client.mean(&sq, &[0], false)?;
    let rms: Vec<f64> = mean_sq.to_vec();
    Ok(rms[0].sqrt())
}

/// Compute normalized error excluding algebraic variables.
fn compute_error_with_exclusion<R, C>(
    client: &C,
    y_new: &Tensor<R>,
    error: &Tensor<R>,
    y_old: &Tensor<R>,
    rtol: f64,
    atol: f64,
    var_types: &Option<Vec<DAEVariableType>>,
) -> Result<f64>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let Some(types) = var_types else {
        return compute_error(client, y_new, error, y_old, rtol, atol);
    };

    let n = y_new.shape()[0];

    // Create mask: 1 for differential, 0 for algebraic
    let mask: Vec<f64> = types
        .iter()
        .map(|t| {
            if *t == DAEVariableType::Differential {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let mask_tensor = Tensor::<R>::from_slice(&mask, &[n], client.device());

    // Mask the error
    let masked_error = client.mul(error, &mask_tensor)?;

    // Count differential variables
    let n_diff = types
        .iter()
        .filter(|t| **t == DAEVariableType::Differential)
        .count() as f64;

    if n_diff < 1.0 {
        // All algebraic - just check that error is small
        let err_norm = compute_norm_scalar(client, error, 2.0)?;
        return Ok(err_norm / atol.max(1e-10));
    }

    // Compute error using only differential variables
    let y_abs = client.abs(y_new)?;
    let y_old_abs = client.abs(y_old)?;
    let max_abs = client.maximum(&y_abs, &y_old_abs)?;
    let scale = client.add_scalar(&client.mul_scalar(&max_abs, rtol)?, atol)?;
    let normalized = client.div(&masked_error, &scale)?;
    let sq = client.mul(&normalized, &normalized)?;
    let sum_sq = client.sum(&sq, &[0], false)?;
    let sum_val: Vec<f64> = sum_sq.to_vec();

    Ok((sum_val[0] / n_diff).sqrt())
}

/// Compute step size adjustment factor.
fn compute_step_factor(error: f64, order: usize) -> f64 {
    if error == 0.0 {
        return MAX_FACTOR;
    }
    let factor = SAFETY * error.powf(-1.0 / (order as f64 + 1.0));
    factor.clamp(MIN_FACTOR, MAX_FACTOR)
}

/// Adjust BDF order based on error.
fn adjust_order(current_order: usize, error: f64, max_order: usize, history_len: usize) -> usize {
    let max_possible = (history_len - 1).min(max_order);
    if current_order >= max_possible {
        return current_order;
    }
    if error < 0.01 && current_order < max_possible {
        current_order + 1
    } else if error > 0.5 && current_order > 1 {
        current_order - 1
    } else {
        current_order
    }
}

/// Build the DAE result tensor.
#[allow(clippy::too_many_arguments)]
fn build_dae_result<R, C>(
    client: &C,
    t_values: &[f64],
    y_values: &[Tensor<R>],
    yp_values: &[Tensor<R>],
    success: bool,
    message: Option<String>,
    nfev: usize,
    njac: usize,
    n_ic_iter: usize,
    naccept: usize,
    nreject: usize,
    return_yp: bool,
) -> IntegrateResult<DAEResultTensor<R>>
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

    let yp_tensor = if return_yp && !yp_values.is_empty() {
        let yp_refs: Vec<&Tensor<R>> = yp_values.iter().collect();
        Some(
            client
                .stack(&yp_refs, 0)
                .map_err(|e| IntegrateError::InvalidInput {
                    context: format!("Failed to stack yp tensors: {}", e),
                })?,
        )
    } else {
        None
    };

    Ok(DAEResultTensor {
        t: t_tensor,
        y: y_tensor,
        yp: yp_tensor,
        success,
        message,
        nfev,
        njac,
        n_ic_iter,
        naccept,
        nreject,
    })
}

fn to_integrate_err(e: numr::error::Error) -> IntegrateError {
    IntegrateError::InvalidInput {
        context: format!("Tensor operation error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::dual_ops::{dual_mul_scalar, dual_sub};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_dae_simple_ode() {
        let (device, client) = setup();

        // dy/dt = -y as DAE: F(t, y, y') = y' + y = 0
        // y(0) = 1, exact: y(t) = exp(-t)
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);
        let yp0 = Tensor::<CpuRuntime>::from_slice(&[-1.0], &[1], &device);

        let f = |_t: &DualTensor<CpuRuntime>,
                 y: &DualTensor<CpuRuntime>,
                 yp: &DualTensor<CpuRuntime>,
                 c: &CpuClient| {
            // F = y' + y (note: this gives ∂F/∂y=1, ∂F/∂y'=1)
            use numr::autograd::dual_ops::dual_add;
            dual_add(yp, y, c)
        };

        let mut ode_opts = ODEOptions::with_tolerances(1e-2, 1e-4);
        ode_opts.h0 = Some(0.1);
        ode_opts.min_step = Some(1e-10);
        let dae_opts = DAEOptions::<CpuRuntime>::default().with_newton_params(1e-4, 50);

        let result = dae_impl(&client, f, [0.0, 2.0], &y0, &yp0, &ode_opts, &dae_opts).unwrap();

        assert!(result.success, "DAE solve failed: {:?}", result.message);

        let y_final = result.y_final_vec();
        let exact = (-2.0_f64).exp();
        assert!(
            (y_final[0] - exact).abs() < 0.1,
            "y_final = {}, exact = {}",
            y_final[0],
            exact
        );
    }

    #[test]
    fn test_dae_stiff_decay() {
        let (device, client) = setup();

        // Stiff: dy/dt = -1000*y as DAE
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);
        let yp0 = Tensor::<CpuRuntime>::from_slice(&[-1000.0], &[1], &device);

        let f = |_t: &DualTensor<CpuRuntime>,
                 y: &DualTensor<CpuRuntime>,
                 yp: &DualTensor<CpuRuntime>,
                 c: &CpuClient| {
            // F = y' + 1000*y
            let neg_1000_y = dual_mul_scalar(y, -1000.0, c)?;
            dual_sub(yp, &neg_1000_y, c)
        };

        let ode_opts = ODEOptions::with_tolerances(1e-4, 1e-6);
        let dae_opts = DAEOptions::<CpuRuntime>::default();

        let result = dae_impl(&client, f, [0.0, 0.05], &y0, &yp0, &ode_opts, &dae_opts).unwrap();

        assert!(result.success, "Stiff DAE failed: {:?}", result.message);

        let y_final = result.y_final_vec();
        // After t=0.05, exp(-1000*0.05) ≈ 1.93e-22
        // Due to tolerances, we just check it's very small
        assert!(
            y_final[0].abs() < 1e-6,
            "y_final = {}, should be ~0",
            y_final[0]
        );
    }

    #[test]
    fn test_dae_with_return_yp() {
        let (device, client) = setup();

        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device);
        let yp0 = Tensor::<CpuRuntime>::from_slice(&[-1.0], &[1], &device);

        let f = |_t: &DualTensor<CpuRuntime>,
                 y: &DualTensor<CpuRuntime>,
                 yp: &DualTensor<CpuRuntime>,
                 c: &CpuClient| { dual_sub(yp, &dual_mul_scalar(y, -1.0, c)?, c) };

        let ode_opts = ODEOptions::with_tolerances(1e-4, 1e-6);
        let dae_opts = DAEOptions::<CpuRuntime>::default().with_return_yp(true);

        let result = dae_impl(&client, f, [0.0, 1.0], &y0, &yp0, &ode_opts, &dae_opts).unwrap();

        assert!(result.yp.is_some(), "yp should be returned");
        let yp_final = result.yp_final_vec().unwrap();

        // At t=1, y ≈ e^-1 ≈ 0.368, so y' ≈ -y ≈ -0.368
        let y_final = result.y_final_vec();
        assert!(
            (yp_final[0] + y_final[0]).abs() < 0.1,
            "y' should equal -y: y'={}, y={}",
            yp_final[0],
            y_final[0]
        );
    }
}
