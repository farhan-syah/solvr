//! Trapezoidal rule integration.

use numr::error::{Error, Result};
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Trapezoidal rule integration using tensors.
///
/// Computes ∫y dx using the composite trapezoidal rule.
/// For 1D tensors, returns a 0-D tensor with the integral.
/// For 2D tensors, integrates each row and returns a 1D tensor.
pub fn trapezoid_impl<R, C>(client: &C, y: &Tensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let y_shape = y.shape();
    let x_shape = x.shape();

    if y_shape.is_empty() || x_shape.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "y/x",
            reason: "trapezoid: tensors must be at least 1D".to_string(),
        });
    }

    let n = y_shape[y_shape.len() - 1];
    let x_n = x_shape[x_shape.len() - 1];

    if n != x_n {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: format!(
                "trapezoid: x and y must have same length in last dimension (got {} and {})",
                x_n, n
            ),
        });
    }

    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: "trapezoid: need at least 2 points".to_string(),
        });
    }

    // Get data as vectors for computation
    // Note: This is acceptable because trapezoid is typically called once
    // with pre-computed y values, not in a loop
    let y_data: Vec<f64> = y.to_vec();
    let x_data: Vec<f64> = x.to_vec();

    // For 1D case
    if y_shape.len() == 1 {
        let mut integral = 0.0;
        for i in 0..n - 1 {
            let dx = x_data[i + 1] - x_data[i];
            integral += 0.5 * dx * (y_data[i] + y_data[i + 1]);
        }
        return Ok(Tensor::<R>::from_slice(&[integral], &[], client.device()));
    }

    // For 2D case (batch integration - integrate each row)
    let batch_size = y_shape[0];
    let mut results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let mut integral = 0.0;
        let row_offset = b * n;
        for i in 0..n - 1 {
            let dx = x_data[i + 1] - x_data[i];
            integral += 0.5 * dx * (y_data[row_offset + i] + y_data[row_offset + i + 1]);
        }
        results.push(integral);
    }

    Ok(Tensor::<R>::from_slice(
        &results,
        &[batch_size],
        client.device(),
    ))
}

/// Trapezoidal rule with uniform spacing.
pub fn trapezoid_uniform_impl<R, C>(client: &C, y: &Tensor<R>, dx: f64) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let y_shape = y.shape();

    if y_shape.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: "trapezoid_uniform: tensor must be at least 1D".to_string(),
        });
    }

    let n = y_shape[y_shape.len() - 1];

    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: "trapezoid_uniform: need at least 2 points".to_string(),
        });
    }

    let y_data: Vec<f64> = y.to_vec();

    // For 1D case
    if y_shape.len() == 1 {
        let mut integral = 0.5 * dx * (y_data[0] + y_data[n - 1]);
        for &val in &y_data[1..n - 1] {
            integral += dx * val;
        }
        return Ok(Tensor::<R>::from_slice(&[integral], &[], client.device()));
    }

    // For 2D case (batch integration)
    let batch_size = y_shape[0];
    let mut results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let row_offset = b * n;
        let mut integral = 0.5 * dx * (y_data[row_offset] + y_data[row_offset + n - 1]);
        for i in 1..n - 1 {
            integral += dx * y_data[row_offset + i];
        }
        results.push(integral);
    }

    Ok(Tensor::<R>::from_slice(
        &results,
        &[batch_size],
        client.device(),
    ))
}

/// Cumulative trapezoidal integration.
///
/// Returns a tensor of the same shape as y with cumulative integrals.
pub fn cumulative_trapezoid_impl<R, C>(
    client: &C,
    y: &Tensor<R>,
    x: Option<&Tensor<R>>,
    dx: f64,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let y_shape = y.shape();

    if y_shape.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: "cumulative_trapezoid: tensor must be at least 1D".to_string(),
        });
    }

    let n = y_shape[y_shape.len() - 1];

    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: "cumulative_trapezoid: need at least 2 points".to_string(),
        });
    }

    let y_data: Vec<f64> = y.to_vec();
    let x_data: Option<Vec<f64>> = x.map(|t| t.to_vec());

    // For 1D case
    if y_shape.len() == 1 {
        let mut result = vec![0.0; n - 1];
        let mut cumsum = 0.0;

        for i in 0..n - 1 {
            let step_dx = if let Some(ref xd) = x_data {
                xd[i + 1] - xd[i]
            } else {
                dx
            };
            cumsum += 0.5 * step_dx * (y_data[i] + y_data[i + 1]);
            result[i] = cumsum;
        }

        return Ok(Tensor::<R>::from_slice(&result, &[n - 1], client.device()));
    }

    // For 2D case
    let batch_size = y_shape[0];
    let out_n = n - 1;
    let mut result = vec![0.0; batch_size * out_n];

    for b in 0..batch_size {
        let row_offset = b * n;
        let out_offset = b * out_n;
        let mut cumsum = 0.0;

        for i in 0..n - 1 {
            let step_dx = if let Some(ref xd) = x_data {
                xd[i + 1] - xd[i]
            } else {
                dx
            };
            cumsum += 0.5 * step_dx * (y_data[row_offset + i] + y_data[row_offset + i + 1]);
            result[out_offset + i] = cumsum;
        }
    }

    Ok(Tensor::<R>::from_slice(
        &result,
        &[batch_size, out_n],
        client.device(),
    ))
}
