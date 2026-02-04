//! Signal decimation with anti-aliasing filter.

use super::helpers::{apply_butter_lowpass, apply_fir_lowpass};
use crate::signal::traits::analysis::{DecimateFilterImpl, DecimateParams};
use numr::algorithm::fft::FftAlgorithms;
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Downsample after applying an anti-aliasing filter.
pub fn decimate_impl<R, C>(
    _client: &C,
    x: &Tensor<R>,
    q: usize,
    params: DecimateParams,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
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

    if q == 0 {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: "Decimation factor must be positive".to_string(),
        });
    }

    if q == 1 {
        return Ok(x.clone());
    }

    // Design anti-aliasing filter
    // Cutoff at 0.8 * Nyquist / q to prevent aliasing
    let cutoff = 0.8 / q as f64;

    // Apply filter
    let filtered = match params.ftype {
        DecimateFilterImpl::Iir => {
            // Use simple IIR Butterworth lowpass
            apply_butter_lowpass(&x_data, cutoff, params.n, params.zero_phase)
        }
        DecimateFilterImpl::Fir => {
            // Use FIR lowpass
            let fir_len = params.n * 2 * q + 1;
            apply_fir_lowpass(&x_data, cutoff, fir_len)
        }
    };

    // Downsample by taking every q-th sample
    let output_len = (n + q - 1) / q;
    let mut result = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let idx = i * q;
        if idx < filtered.len() {
            result.push(filtered[idx]);
        }
    }

    Ok(Tensor::from_slice(&result, &[result.len()], device))
}
