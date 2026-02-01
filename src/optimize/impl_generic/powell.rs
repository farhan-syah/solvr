//! Powell's method for multivariate minimization.

use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::MinimizeOptions;

use super::helpers::{TensorMinimizeResult, line_search_quadratic};
use super::utils::SINGULAR_THRESHOLD;

/// Powell's method for minimization using tensors.
pub fn powell_impl<R, C, F>(
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
            context: "powell: empty initial guess".to_string(),
        });
    }

    let mut x: Vec<f64> = x0.to_vec();
    let x_tensor = Tensor::<R>::from_slice(&x, &[n], client.device());
    let mut fx = f(&x_tensor).map_err(|e| OptimizeError::NumericalError {
        message: format!("powell: initial evaluation - {}", e),
    })?;
    let mut nfev = 1;

    // Initialize direction set to identity
    let mut directions: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut d = vec![0.0; n];
            d[i] = 1.0;
            d
        })
        .collect();

    for iter in 0..options.max_iter {
        let x_start = x.clone();
        let fx_start = fx;

        let mut max_decrease = 0.0;
        let mut max_decrease_idx = 0;

        // Line search along each direction
        for (i, direction) in directions.iter().enumerate() {
            let (x_new, fx_new, evals) =
                line_search_quadratic::<R, C, F>(client, &f, &x, direction, fx, n)?;
            nfev += evals;

            let decrease = fx - fx_new;
            if decrease > max_decrease {
                max_decrease = decrease;
                max_decrease_idx = i;
            }

            x = x_new;
            fx = fx_new;
        }

        // Check convergence
        if 2.0 * (fx_start - fx).abs()
            <= options.f_tol * (fx_start.abs() + fx.abs() + SINGULAR_THRESHOLD)
        {
            let x_result = Tensor::<R>::from_slice(&x, &[n], client.device());
            return Ok(TensorMinimizeResult {
                x: x_result,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Update direction set
        let new_direction: Vec<f64> = x.iter().zip(x_start.iter()).map(|(a, b)| a - b).collect();
        let new_dir_norm: f64 = new_direction.iter().map(|d| d * d).sum::<f64>().sqrt();

        if new_dir_norm > SINGULAR_THRESHOLD {
            directions.remove(max_decrease_idx);
            directions.push(new_direction);
        }
    }

    let x_result = Tensor::<R>::from_slice(&x, &[n], client.device());
    Ok(TensorMinimizeResult {
        x: x_result,
        fun: fx,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}
