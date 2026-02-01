//! Shared helper functions and types for multivariate minimization.

use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};

use super::utils::tensor_dot;

/// Result of tensor-based minimization.
#[derive(Debug, Clone)]
pub struct TensorMinimizeResult<R: Runtime> {
    /// Solution vector.
    pub x: Tensor<R>,
    /// Function value at solution.
    pub fun: f64,
    /// Number of iterations.
    pub iterations: usize,
    /// Number of function evaluations.
    pub nfev: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

/// Backtracking line search with Armijo condition.
pub fn backtracking_line_search_tensor<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    p: &Tensor<R>,
    fx: f64,
    grad: &Tensor<R>,
) -> OptimizeResult<(Tensor<R>, f64, usize)>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let c = 0.0001;
    let rho = 0.5;
    let n = x.shape()[0];

    let grad_dot_p = tensor_dot(client, grad, p).map_err(|e| OptimizeError::NumericalError {
        message: format!("line_search: grad_dot_p - {}", e),
    })?;

    let x_data: Vec<f64> = x.to_vec();
    let p_data: Vec<f64> = p.to_vec();

    let mut alpha = 1.0;
    let mut nfev = 0;

    for _ in 0..50 {
        let x_new_data: Vec<f64> = x_data
            .iter()
            .zip(p_data.iter())
            .map(|(xi, pi)| xi + alpha * pi)
            .collect();
        let x_new = Tensor::<R>::from_slice(&x_new_data, &[n], client.device());
        let fx_new = f(&x_new).map_err(|e| OptimizeError::NumericalError {
            message: format!("line_search: f eval - {}", e),
        })?;
        nfev += 1;

        if fx_new <= fx + c * alpha * grad_dot_p {
            return Ok((x_new, fx_new, nfev));
        }

        alpha *= rho;
    }

    Ok((x.clone(), fx, nfev))
}

/// Quadratic line search for Powell's method.
pub fn line_search_quadratic<R, C, F>(
    client: &C,
    f: &F,
    x: &[f64],
    direction: &[f64],
    fx: f64,
    n: usize,
) -> OptimizeResult<(Vec<f64>, f64, usize)>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let mut alpha = 0.1;
    let mut nfev = 0;

    let mut best_x = x.to_vec();
    let mut best_fx = fx;

    for _ in 0..20 {
        let x_new: Vec<f64> = x
            .iter()
            .zip(direction.iter())
            .map(|(xi, di)| xi + alpha * di)
            .collect();
        let x_tensor = Tensor::<R>::from_slice(&x_new, &[n], client.device());
        let fx_new = f(&x_tensor).map_err(|e| OptimizeError::NumericalError {
            message: format!("powell line_search - {}", e),
        })?;
        nfev += 1;

        if fx_new < best_fx {
            best_x = x_new;
            best_fx = fx_new;
            alpha *= 1.5;
        } else {
            alpha *= 0.5;
            if alpha < 1e-10 {
                break;
            }
        }
    }

    Ok((best_x, best_fx, nfev))
}

/// Compare two f64 values, treating NaN as greater than all other values.
/// This ensures NaN values sort to the end.
pub fn compare_f64_nan_safe(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| {
        // If comparison fails (one or both is NaN):
        // - NaN should sort to the end (be "greater")
        match (a.is_nan(), b.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => std::cmp::Ordering::Equal, // unreachable
        }
    })
}
