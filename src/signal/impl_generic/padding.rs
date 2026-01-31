//! Generic padding operations for signal processing.
//!
//! These functions handle zero-padding and reflect-padding for FFT-based
//! convolution and STFT operations.

use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Pad 1D tensor to specified length - zero-padding at end (generic over Runtime).
pub fn pad_1d_to_length_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    target_len: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let dtype = tensor.dtype();
    let tensor_contig = tensor.contiguous();
    let ndim = tensor_contig.ndim();
    let current_len = tensor_contig.shape()[ndim - 1];

    if current_len >= target_len {
        return Ok(tensor_contig);
    }

    let mut out_shape: Vec<usize> = tensor_contig.shape().to_vec();
    out_shape[ndim - 1] = target_len;

    let batch_size: usize = tensor_contig.shape()[..ndim - 1].iter().product();
    let batch_size = batch_size.max(1);

    match dtype {
        DType::F32 => {
            let data: Vec<f32> = tensor_contig.to_vec();
            let mut output = vec![0.0f32; batch_size * target_len];

            for b in 0..batch_size {
                let src_offset = b * current_len;
                let dst_offset = b * target_len;
                output[dst_offset..dst_offset + current_len]
                    .copy_from_slice(&data[src_offset..src_offset + current_len]);
            }

            Ok(Tensor::<R>::from_slice(
                &output,
                &out_shape,
                client.device(),
            ))
        }
        DType::F64 => {
            let data: Vec<f64> = tensor_contig.to_vec();
            let mut output = vec![0.0f64; batch_size * target_len];

            for b in 0..batch_size {
                let src_offset = b * current_len;
                let dst_offset = b * target_len;
                output[dst_offset..dst_offset + current_len]
                    .copy_from_slice(&data[src_offset..src_offset + current_len]);
            }

            Ok(Tensor::<R>::from_slice(
                &output,
                &out_shape,
                client.device(),
            ))
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "pad_1d_to_length",
        }),
    }
}

/// Pad 2D tensor to specified shape - zero-padding (generic over Runtime).
pub fn pad_2d_to_shape_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    target_h: usize,
    target_w: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let dtype = tensor.dtype();
    let tensor_contig = tensor.contiguous();
    let ndim = tensor_contig.ndim();
    let current_h = tensor_contig.shape()[ndim - 2];
    let current_w = tensor_contig.shape()[ndim - 1];

    let mut out_shape: Vec<usize> = tensor_contig.shape().to_vec();
    out_shape[ndim - 2] = target_h;
    out_shape[ndim - 1] = target_w;

    let batch_size: usize = tensor_contig.shape()[..ndim - 2].iter().product();
    let batch_size = batch_size.max(1);

    match dtype {
        DType::F32 => {
            let data: Vec<f32> = tensor_contig.to_vec();
            let mut output = vec![0.0f32; batch_size * target_h * target_w];

            for b in 0..batch_size {
                for row in 0..current_h {
                    let src_offset = b * current_h * current_w + row * current_w;
                    let dst_offset = b * target_h * target_w + row * target_w;
                    output[dst_offset..dst_offset + current_w]
                        .copy_from_slice(&data[src_offset..src_offset + current_w]);
                }
            }

            Ok(Tensor::<R>::from_slice(
                &output,
                &out_shape,
                client.device(),
            ))
        }
        DType::F64 => {
            let data: Vec<f64> = tensor_contig.to_vec();
            let mut output = vec![0.0f64; batch_size * target_h * target_w];

            for b in 0..batch_size {
                for row in 0..current_h {
                    let src_offset = b * current_h * current_w + row * current_w;
                    let dst_offset = b * target_h * target_w + row * target_w;
                    output[dst_offset..dst_offset + current_w]
                        .copy_from_slice(&data[src_offset..src_offset + current_w]);
                }
            }

            Ok(Tensor::<R>::from_slice(
                &output,
                &out_shape,
                client.device(),
            ))
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "pad_2d_to_shape",
        }),
    }
}

/// Reflect padding for 1D signal - used in STFT centering (generic over Runtime).
pub fn pad_1d_reflect_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    pad_left: usize,
    pad_right: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let dtype = tensor.dtype();
    let tensor_contig = tensor.contiguous();
    let ndim = tensor_contig.ndim();
    let current_len = tensor_contig.shape()[ndim - 1];

    let target_len = current_len + pad_left + pad_right;
    let mut out_shape: Vec<usize> = tensor_contig.shape().to_vec();
    out_shape[ndim - 1] = target_len;

    let batch_size: usize = tensor_contig.shape()[..ndim - 1].iter().product();
    let batch_size = batch_size.max(1);

    match dtype {
        DType::F32 => {
            let data: Vec<f32> = tensor_contig.to_vec();
            let mut output = vec![0.0f32; batch_size * target_len];

            for b in 0..batch_size {
                let src_offset = b * current_len;
                let dst_offset = b * target_len;
                let src = &data[src_offset..src_offset + current_len];
                let dst = &mut output[dst_offset..dst_offset + target_len];

                reflect_pad_1d(src, dst, pad_left, current_len);
            }

            Ok(Tensor::<R>::from_slice(
                &output,
                &out_shape,
                client.device(),
            ))
        }
        DType::F64 => {
            let data: Vec<f64> = tensor_contig.to_vec();
            let mut output = vec![0.0f64; batch_size * target_len];

            for b in 0..batch_size {
                let src_offset = b * current_len;
                let dst_offset = b * target_len;
                let src = &data[src_offset..src_offset + current_len];
                let dst = &mut output[dst_offset..dst_offset + target_len];

                reflect_pad_1d(src, dst, pad_left, current_len);
            }

            Ok(Tensor::<R>::from_slice(
                &output,
                &out_shape,
                client.device(),
            ))
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "pad_1d_reflect",
        }),
    }
}

/// Core reflect padding algorithm.
fn reflect_pad_1d<T: Copy>(src: &[T], dst: &mut [T], pad_left: usize, current_len: usize) {
    let target_len = dst.len();

    // Left padding (reflected)
    for (i, dst_val) in dst.iter_mut().take(pad_left).enumerate() {
        let reflect_idx = (pad_left - i).min(current_len - 1);
        *dst_val = src[reflect_idx];
    }

    // Original data
    dst[pad_left..pad_left + current_len].copy_from_slice(src);

    // Right padding (reflected)
    for i in 0..(target_len - pad_left - current_len) {
        let idx = if current_len <= 1 {
            0
        } else {
            let period = current_len - 1;
            let pos = i % (2 * period);
            if pos < period {
                current_len - 2 - pos
            } else {
                pos - period + 1
            }
        };
        dst[pad_left + current_len + i] = src[idx];
    }
}
