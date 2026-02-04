//! Savitzky-Golay filter implementation.

use super::helpers::compute_savgol_coeffs;
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Apply Savitzky-Golay filter.
pub fn savgol_filter_impl<R, C>(
    _client: &C,
    x: &Tensor<R>,
    window_length: usize,
    polyorder: usize,
    deriv: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let x_data: Vec<f64> = x.to_vec();
    let n = x_data.len();
    let device = x.device();

    if window_length % 2 == 0 {
        return Err(Error::InvalidArgument {
            arg: "window_length",
            reason: "window_length must be odd".to_string(),
        });
    }

    if window_length < polyorder + 2 {
        return Err(Error::InvalidArgument {
            arg: "window_length",
            reason: "window_length must be greater than polyorder + 1".to_string(),
        });
    }

    if deriv > polyorder {
        return Err(Error::InvalidArgument {
            arg: "deriv",
            reason: "deriv must be <= polyorder".to_string(),
        });
    }

    let half_window = window_length / 2;

    // Compute Savitzky-Golay coefficients
    let coeffs = compute_savgol_coeffs(window_length, polyorder, deriv);

    // Apply filter
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let mut sum = 0.0;

        for j in 0..window_length {
            let k = i as isize + j as isize - half_window as isize;
            // Mirror boundary conditions
            let idx = if k < 0 {
                (-k) as usize
            } else if k >= n as isize {
                2 * n - 2 - k as usize
            } else {
                k as usize
            };

            if idx < n {
                sum += coeffs[j] * x_data[idx];
            }
        }

        result.push(sum);
    }

    Ok(Tensor::from_slice(&result, &[n], device))
}
