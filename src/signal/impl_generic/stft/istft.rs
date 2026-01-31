//! Generic ISTFT (Inverse Short-Time Fourier Transform) implementation.

#![allow(clippy::too_many_arguments)]

use crate::signal::validate_stft_params;
use crate::window::WindowFunctions;
use numr::algorithm::fft::{FftAlgorithms, FftNormalization};
use numr::dtype::{Complex64, Complex128, DType};
use numr::error::{Error, Result};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generic implementation of ISTFT.
pub fn istft_impl<R, C>(
    client: &C,
    stft_matrix: &Tensor<R>,
    hop_length: Option<usize>,
    window: Option<&Tensor<R>>,
    center: bool,
    length: Option<usize>,
    normalized: bool,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + WindowFunctions<R> + RuntimeClient<R>,
{
    let dtype = stft_matrix.dtype();

    if !dtype.is_complex() {
        return Err(Error::UnsupportedDType { dtype, op: "istft" });
    }

    let real_dtype = match dtype {
        DType::Complex64 => DType::F32,
        DType::Complex128 => DType::F64,
        _ => unreachable!(),
    };

    let stft_contig = stft_matrix.contiguous();
    let ndim = stft_contig.ndim();

    if ndim < 2 {
        return Err(Error::InvalidArgument {
            arg: "stft_matrix",
            reason: "istft requires at least 2D input [n_frames, freq_bins]".to_string(),
        });
    }

    let n_frames = stft_contig.shape()[ndim - 2];
    let freq_bins = stft_contig.shape()[ndim - 1];
    let n_fft = (freq_bins - 1) * 2;

    let hop = hop_length.unwrap_or(n_fft / 4);
    validate_stft_params(n_fft, hop, "istft")?;

    // Get or create window
    let default_window;
    let win = if let Some(w) = window {
        if w.shape() != [n_fft] {
            return Err(Error::InvalidArgument {
                arg: "window",
                reason: format!("window must have shape [{n_fft}], got {:?}", w.shape()),
            });
        }
        w
    } else {
        default_window = client.hann_window(n_fft, real_dtype, client.device())?;
        &default_window
    };

    let batch_size: usize = if ndim > 2 {
        stft_contig.shape()[..ndim - 2].iter().product()
    } else {
        1
    };

    // Calculate output length
    let expected_len = n_fft + (n_frames - 1) * hop;
    let output_len = if center {
        expected_len - n_fft // Remove padding
    } else {
        expected_len
    };
    let final_len = length.unwrap_or(output_len);

    // Output shape: [..., final_len]
    let mut out_shape: Vec<usize> = stft_contig.shape()[..ndim - 2].to_vec();
    out_shape.push(final_len);

    let norm = if normalized {
        FftNormalization::Ortho
    } else {
        FftNormalization::Backward
    };

    match real_dtype {
        DType::F32 => istft_process_f32(
            client,
            &stft_contig,
            win,
            &out_shape,
            n_fft,
            hop,
            n_frames,
            batch_size,
            freq_bins,
            center,
            final_len,
            norm,
        ),
        DType::F64 => istft_process_f64(
            client,
            &stft_contig,
            win,
            &out_shape,
            n_fft,
            hop,
            n_frames,
            batch_size,
            freq_bins,
            center,
            final_len,
            norm,
        ),
        _ => Err(Error::UnsupportedDType {
            dtype: real_dtype,
            op: "istft",
        }),
    }
}

fn istft_process_f32<R, C>(
    client: &C,
    stft_matrix: &Tensor<R>,
    window: &Tensor<R>,
    out_shape: &[usize],
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    batch_size: usize,
    freq_bins: usize,
    center: bool,
    final_len: usize,
    norm: FftNormalization,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + RuntimeClient<R>,
{
    let stft_data: Vec<Complex64> = stft_matrix.to_vec();
    let window_data: Vec<f32> = window.to_vec();

    let full_len = n_fft + (n_frames - 1) * hop;
    let pad_left = if center { n_fft / 2 } else { 0 };

    let mut output_data = vec![0.0f32; batch_size * final_len];

    for b in 0..batch_size {
        let stft_offset = b * n_frames * freq_bins;
        let out_offset = b * final_len;

        let mut reconstruction = vec![0.0f32; full_len];
        let mut window_sum = vec![0.0f32; full_len];

        for f in 0..n_frames {
            let frame_spectrum: Vec<Complex64> =
                stft_data[stft_offset + f * freq_bins..stft_offset + (f + 1) * freq_bins].to_vec();

            let spectrum_tensor =
                Tensor::<R>::from_slice(&frame_spectrum, &[freq_bins], client.device());
            let frame = client.irfft(&spectrum_tensor, Some(n_fft), norm)?;
            let frame_data: Vec<f32> = frame.to_vec();

            let frame_start = f * hop;
            for i in 0..n_fft {
                let out_idx = frame_start + i;
                if out_idx < full_len {
                    let win_val = window_data[i];
                    reconstruction[out_idx] += frame_data[i] * win_val;
                    window_sum[out_idx] += win_val * win_val;
                }
            }
        }

        // Normalize and copy to output
        for i in 0..final_len {
            let src_idx = pad_left + i;
            if src_idx < full_len {
                let norm_factor = if window_sum[src_idx] > 1e-8 {
                    window_sum[src_idx]
                } else {
                    1.0
                };
                output_data[out_offset + i] = reconstruction[src_idx] / norm_factor;
            }
        }
    }

    Ok(Tensor::<R>::from_slice(
        &output_data,
        out_shape,
        client.device(),
    ))
}

fn istft_process_f64<R, C>(
    client: &C,
    stft_matrix: &Tensor<R>,
    window: &Tensor<R>,
    out_shape: &[usize],
    n_fft: usize,
    hop: usize,
    n_frames: usize,
    batch_size: usize,
    freq_bins: usize,
    center: bool,
    final_len: usize,
    norm: FftNormalization,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: FftAlgorithms<R> + RuntimeClient<R>,
{
    let stft_data: Vec<Complex128> = stft_matrix.to_vec();
    let window_data: Vec<f64> = window.to_vec();

    let full_len = n_fft + (n_frames - 1) * hop;
    let pad_left = if center { n_fft / 2 } else { 0 };

    let mut output_data = vec![0.0f64; batch_size * final_len];

    for b in 0..batch_size {
        let stft_offset = b * n_frames * freq_bins;
        let out_offset = b * final_len;

        let mut reconstruction = vec![0.0f64; full_len];
        let mut window_sum = vec![0.0f64; full_len];

        for f in 0..n_frames {
            let frame_spectrum: Vec<Complex128> =
                stft_data[stft_offset + f * freq_bins..stft_offset + (f + 1) * freq_bins].to_vec();

            let spectrum_tensor =
                Tensor::<R>::from_slice(&frame_spectrum, &[freq_bins], client.device());
            let frame = client.irfft(&spectrum_tensor, Some(n_fft), norm)?;
            let frame_data: Vec<f64> = frame.to_vec();

            let frame_start = f * hop;
            for i in 0..n_fft {
                let out_idx = frame_start + i;
                if out_idx < full_len {
                    let win_val = window_data[i];
                    reconstruction[out_idx] += frame_data[i] * win_val;
                    window_sum[out_idx] += win_val * win_val;
                }
            }
        }

        for i in 0..final_len {
            let src_idx = pad_left + i;
            if src_idx < full_len {
                let norm_factor = if window_sum[src_idx] > 1e-8 {
                    window_sum[src_idx]
                } else {
                    1.0
                };
                output_data[out_offset + i] = reconstruction[src_idx] / norm_factor;
            }
        }
    }

    Ok(Tensor::<R>::from_slice(
        &output_data,
        out_shape,
        client.device(),
    ))
}
