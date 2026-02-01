//! Powell's method for multivariate minimization.

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::MinimizeOptions;

use super::helpers::{TensorMinimizeResult, line_search_tensor};
use super::utils::{SINGULAR_THRESHOLD, tensor_norm};

/// Powell's method for minimization using tensors.
///
/// All operations use tensor ops to stay on device (CPU/GPU).
pub fn powell_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    options: &MinimizeOptions,
) -> OptimizeResult<TensorMinimizeResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let n = x0.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "powell: empty initial guess".to_string(),
        });
    }

    let mut x = x0.clone();
    let mut fx = f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("powell: initial evaluation - {}", e),
    })?;
    let mut nfev = 1;

    // Initialize direction set to identity matrix [n, n]
    // directions[i, :] is the i-th search direction
    let mut directions = create_identity_matrix::<R, C>(client, n)?;

    for iter in 0..options.max_iter {
        let x_start = x.clone();
        let fx_start = fx;

        let mut max_decrease = 0.0;
        let mut max_decrease_idx = 0;

        // Line search along each direction
        for i in 0..n {
            // Extract i-th direction (row i of directions matrix)
            let direction = extract_row(client, &directions, i)?;

            // Line search along this direction
            let (x_new, fx_new, evals) =
                line_search_tensor(client, &f, &x, &direction, fx)?;
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
            return Ok(TensorMinimizeResult {
                x,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute new direction: x - x_start
        let new_direction = client
            .sub(&x, &x_start)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("powell: new direction - {}", e),
            })?;
        let new_dir_norm =
            tensor_norm(client, &new_direction).map_err(|e| OptimizeError::NumericalError {
                message: format!("powell: direction norm - {}", e),
            })?;

        if new_dir_norm > SINGULAR_THRESHOLD {
            // Update direction set: shift directions and add new one
            directions = update_direction_set(client, &directions, max_decrease_idx, &new_direction, n)?;
        }
    }

    Ok(TensorMinimizeResult {
        x,
        fun: fx,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

/// Create an n x n identity matrix as a tensor.
fn create_identity_matrix<R, C>(client: &C, n: usize) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let mut data = vec![0.0f64; n * n];
    for i in 0..n {
        data[i * n + i] = 1.0;
    }
    Ok(Tensor::<R>::from_slice(&data, &[n, n], client.device()))
}

/// Extract row i from a 2D tensor as a 1D tensor.
fn extract_row<R, C>(client: &C, matrix: &Tensor<R>, row: usize) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let n = matrix.shape()[1];
    // Create index tensor for the row
    let indices = Tensor::<R>::from_slice(&[row as i64], &[1], client.device());
    let row_2d = client
        .index_select(matrix, 0, &indices)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("powell: extract row - {}", e),
        })?;
    // Reshape from [1, n] to [n]
    row_2d
        .reshape(&[n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("powell: reshape row - {}", e),
        })
}

/// Update direction set by removing direction at idx and adding new_direction at the end.
///
/// This implements the Powell direction update: remove the direction that gave
/// the maximum decrease and append the new extrapolated direction.
fn update_direction_set<R, C>(
    client: &C,
    directions: &Tensor<R>,
    remove_idx: usize,
    new_direction: &Tensor<R>,
    n: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    // For simplicity, we reconstruct the matrix
    // This is acceptable as direction update is O(n²) anyway and done once per iteration
    let dir_data: Vec<f64> = directions.to_vec();
    let new_dir_data: Vec<f64> = new_direction.to_vec();

    let mut new_data = Vec::with_capacity(n * n);

    // Copy all rows except the one to remove, then append new direction
    for i in 0..n {
        if i == remove_idx {
            continue;
        }
        let start = i * n;
        new_data.extend_from_slice(&dir_data[start..start + n]);
    }
    // Append new direction as last row
    new_data.extend_from_slice(&new_dir_data);

    Ok(Tensor::<R>::from_slice(&new_data, &[n, n], client.device()))
}
