//! BFGS quasi-Newton method for multivariate minimization.

use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::MinimizeOptions;

use super::helpers::{TensorMinimizeResult, backtracking_line_search_tensor};
use super::utils::{SINGULAR_THRESHOLD, finite_difference_gradient, tensor_norm};

/// BFGS quasi-Newton method for minimization using tensors.
pub fn bfgs_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    options: &MinimizeOptions,
) -> OptimizeResult<TensorMinimizeResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let n = x0.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "bfgs: empty initial guess".to_string(),
        });
    }

    let mut x = x0.clone();
    let mut fx = f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("bfgs: initial evaluation - {}", e),
    })?;
    let mut nfev = 1;

    let mut grad = finite_difference_gradient(client, &f, &x, fx, options.eps).map_err(|e| {
        OptimizeError::NumericalError {
            message: format!("bfgs: gradient - {}", e),
        }
    })?;
    nfev += n;

    // Initialize inverse Hessian approximation to identity (stored as Vec for simplicity)
    let mut h_inv: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect();

    for iter in 0..options.max_iter {
        let grad_norm = tensor_norm(client, &grad).map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs: grad norm - {}", e),
        })?;

        if grad_norm < options.g_tol {
            return Ok(TensorMinimizeResult {
                x,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute search direction: p = -H_inv * grad
        let grad_data: Vec<f64> = grad.to_vec();
        let mut p_data = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                p_data[i] -= h_inv[i][j] * grad_data[j];
            }
        }
        let p = Tensor::<R>::from_slice(&p_data, &[n], client.device());

        // Line search
        let (x_new, fx_new, evals) =
            backtracking_line_search_tensor(client, &f, &x, &p, fx, &grad)?;
        nfev += evals;

        // Check convergence
        let dx = client
            .sub(&x_new, &x)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("bfgs: dx - {}", e),
            })?;
        let dx_norm = tensor_norm(client, &dx).map_err(|e| OptimizeError::NumericalError {
            message: format!("bfgs: dx norm - {}", e),
        })?;

        if dx_norm < options.x_tol || (fx - fx_new).abs() < options.f_tol {
            return Ok(TensorMinimizeResult {
                x: x_new,
                fun: fx_new,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute new gradient
        let grad_new = finite_difference_gradient(client, &f, &x_new, fx_new, options.eps)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("bfgs: new gradient - {}", e),
            })?;
        nfev += n;

        // BFGS update
        let s: Vec<f64> = {
            let x_data: Vec<f64> = x.to_vec();
            let x_new_data: Vec<f64> = x_new.to_vec();
            x_new_data
                .iter()
                .zip(x_data.iter())
                .map(|(a, b)| a - b)
                .collect()
        };
        let y: Vec<f64> = {
            let grad_new_data: Vec<f64> = grad_new.to_vec();
            grad_new_data
                .iter()
                .zip(grad_data.iter())
                .map(|(a, b)| a - b)
                .collect()
        };

        let ys: f64 = y.iter().zip(s.iter()).map(|(a, b)| a * b).sum();
        if ys.abs() > SINGULAR_THRESHOLD {
            let rho = 1.0 / ys;

            let mut h_y = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    h_y[i] += h_inv[i][j] * y[j];
                }
            }

            let yhy: f64 = y.iter().zip(h_y.iter()).map(|(a, b)| a * b).sum();

            for i in 0..n {
                for j in 0..n {
                    h_inv[i][j] += rho * (1.0 + rho * yhy) * s[i] * s[j]
                        - rho * (s[i] * h_y[j] + h_y[i] * s[j]);
                }
            }
        }

        x = x_new;
        fx = fx_new;
        grad = grad_new;
    }

    Ok(TensorMinimizeResult {
        x,
        fun: fx,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}
