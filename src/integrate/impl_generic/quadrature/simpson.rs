//! Simpson's rule integration.

use numr::error::{Error, Result};
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Simpson's rule integration.
///
/// Uses Simpson's 1/3 rule for even intervals, with trapezoidal for odd.
pub fn simpson_impl<R, C>(
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
            reason: "simpson: tensor must be at least 1D".to_string(),
        });
    }

    let n = y_shape[y_shape.len() - 1];

    if n < 2 {
        return Err(Error::InvalidArgument {
            arg: "y",
            reason: "simpson: need at least 2 points".to_string(),
        });
    }

    let y_data: Vec<f64> = y.to_vec();
    let x_data: Option<Vec<f64>> = x.map(|t| t.to_vec());

    // For 1D case
    if y_shape.len() == 1 {
        let integral = simpson_1d(&y_data, x_data.as_deref(), dx);
        return Ok(Tensor::<R>::from_slice(&[integral], &[], client.device()));
    }

    // For 2D case (batch integration)
    let batch_size = y_shape[0];
    let mut results = Vec::with_capacity(batch_size);

    for b in 0..batch_size {
        let row_offset = b * n;
        let row = &y_data[row_offset..row_offset + n];
        let integral = simpson_1d(row, x_data.as_deref(), dx);
        results.push(integral);
    }

    Ok(Tensor::<R>::from_slice(
        &results,
        &[batch_size],
        client.device(),
    ))
}

/// Simpson's rule for 1D data.
fn simpson_1d(y: &[f64], x: Option<&[f64]>, dx: f64) -> f64 {
    let n = y.len();

    if n < 2 {
        return 0.0;
    }

    if n == 2 {
        // Just use trapezoidal for 2 points
        let h = x.map_or(dx, |xd| xd[1] - xd[0]);
        return 0.5 * h * (y[0] + y[1]);
    }

    // Number of intervals
    let intervals = n - 1;

    if intervals.is_multiple_of(2) {
        // Even number of intervals - pure Simpson's 1/3 rule
        let mut integral = 0.0;
        for i in (0..intervals).step_by(2) {
            let h = x.map_or(dx, |xd| (xd[i + 2] - xd[i]) / 2.0);
            integral += h / 3.0 * (y[i] + 4.0 * y[i + 1] + y[i + 2]);
        }
        integral
    } else {
        // Odd number of intervals - use Simpson's for most, trapezoidal for last
        let mut integral = 0.0;

        // Simpson's for even part
        for i in (0..intervals - 1).step_by(2) {
            let h = x.map_or(dx, |xd| (xd[i + 2] - xd[i]) / 2.0);
            integral += h / 3.0 * (y[i] + 4.0 * y[i + 1] + y[i + 2]);
        }

        // Trapezoidal for last interval
        let h = x.map_or(dx, |xd| xd[n - 1] - xd[n - 2]);
        integral += 0.5 * h * (y[n - 2] + y[n - 1]);

        integral
    }
}
