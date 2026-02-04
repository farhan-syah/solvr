//! Signal resampling using FFT-based method.
//!
//! Note: This implementation uses CPU-based FFT because FFT-based resampling
//! requires arbitrary-length FFTs (not just power-of-2), which GPU FFT
//! implementations typically don't support.

use super::helpers::{compute_fft, compute_ifft};
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Resample a signal using FFT-based method.
///
/// Note: Uses CPU FFT because resampling requires arbitrary-length FFTs.
/// The spectral algorithms (welch, periodogram, etc.) use backend-optimized
/// rfft because they can pad to power-of-2 sizes.
pub fn resample_impl<R, C>(_client: &C, x: &Tensor<R>, num: usize, den: usize) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let x_data: Vec<f64> = x.to_vec();
    let n = x_data.len();
    let device = x.device();

    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input signal cannot be empty".to_string(),
        });
    }

    if num == 0 || den == 0 {
        return Err(Error::InvalidArgument {
            arg: "num/den",
            reason: "Resampling factors must be positive".to_string(),
        });
    }

    // Simple case: no resampling
    if num == den {
        return Ok(x.clone());
    }

    // For simple cases, use FFT-based resampling
    // This is equivalent to scipy.signal.resample
    let output_len = (n * num + den - 1) / den;

    // Upsample by num (zero-padding in frequency domain)
    // Then downsample by den (truncation in time domain)

    // Compute FFT (CPU-based for arbitrary length support)
    let fft = compute_fft(&x_data);

    // Create new frequency array with target length
    let target_fft_len = output_len;
    let mut new_fft = vec![(0.0, 0.0); target_fft_len];

    // Copy frequencies, handling up/downsampling
    let half_orig = (n + 1) / 2;
    let half_new = (target_fft_len + 1) / 2;
    let copy_len = half_orig.min(half_new);

    // Copy positive frequencies (including DC)
    for i in 0..copy_len {
        new_fft[i] = fft[i];
    }

    // Copy negative frequencies
    if n > 1 && target_fft_len > 1 {
        let neg_copy_len = (n / 2).min(target_fft_len / 2);
        for i in 1..=neg_copy_len {
            if n - i < fft.len() && target_fft_len - i < new_fft.len() {
                new_fft[target_fft_len - i] = fft[n - i];
            }
        }
    }

    // Scale by ratio to preserve amplitude
    let scale = target_fft_len as f64 / n as f64;
    for f in &mut new_fft {
        f.0 *= scale;
        f.1 *= scale;
    }

    // Compute IFFT
    let result = compute_ifft(&new_fft);
    let result_data: Vec<f64> = result.iter().map(|(re, _)| *re).collect();

    Ok(Tensor::from_slice(&result_data, &[output_len], device))
}
