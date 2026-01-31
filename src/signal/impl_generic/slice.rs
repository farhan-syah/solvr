//! Generic slicing operations for signal processing.
//!
//! These functions extract slices from the last dimension(s) of tensors,
//! used for extracting convolution results based on output mode.

use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Slice last dimension of tensor (generic over Runtime).
///
/// Extracts elements [start, start + len) from the last dimension.
pub fn slice_last_dim_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    start: usize,
    len: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let dtype = tensor.dtype();
    let ndim = tensor.ndim();
    let src_stride = tensor.shape()[ndim - 1];

    let mut out_shape: Vec<usize> = tensor.shape().to_vec();
    out_shape[ndim - 1] = len;

    let batch_size: usize = tensor.shape()[..ndim - 1].iter().product();
    let batch_size = batch_size.max(1);

    let tensor_contig = tensor.contiguous();

    match dtype {
        DType::F32 => {
            let data: Vec<f32> = tensor_contig.to_vec();
            let mut output = vec![0.0f32; batch_size * len];

            for b in 0..batch_size {
                let src_offset = b * src_stride + start;
                let dst_offset = b * len;
                output[dst_offset..dst_offset + len]
                    .copy_from_slice(&data[src_offset..src_offset + len]);
            }

            Ok(Tensor::<R>::from_slice(
                &output,
                &out_shape,
                client.device(),
            ))
        }
        DType::F64 => {
            let data: Vec<f64> = tensor_contig.to_vec();
            let mut output = vec![0.0f64; batch_size * len];

            for b in 0..batch_size {
                let src_offset = b * src_stride + start;
                let dst_offset = b * len;
                output[dst_offset..dst_offset + len]
                    .copy_from_slice(&data[src_offset..src_offset + len]);
            }

            Ok(Tensor::<R>::from_slice(
                &output,
                &out_shape,
                client.device(),
            ))
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "slice_last_dim",
        }),
    }
}

/// Slice last two dimensions of tensor (generic over Runtime).
///
/// Extracts a rectangular region from the last two dimensions.
pub fn slice_last_2d_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    start_h: usize,
    len_h: usize,
    start_w: usize,
    len_w: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let dtype = tensor.dtype();
    let ndim = tensor.ndim();
    let src_h = tensor.shape()[ndim - 2];
    let src_w = tensor.shape()[ndim - 1];

    let mut out_shape: Vec<usize> = tensor.shape().to_vec();
    out_shape[ndim - 2] = len_h;
    out_shape[ndim - 1] = len_w;

    let batch_size: usize = tensor.shape()[..ndim - 2].iter().product();
    let batch_size = batch_size.max(1);

    let tensor_contig = tensor.contiguous();

    match dtype {
        DType::F32 => {
            let data: Vec<f32> = tensor_contig.to_vec();
            let mut output = vec![0.0f32; batch_size * len_h * len_w];

            for b in 0..batch_size {
                for row in 0..len_h {
                    let src_row = start_h + row;
                    let src_offset = b * src_h * src_w + src_row * src_w + start_w;
                    let dst_offset = b * len_h * len_w + row * len_w;
                    output[dst_offset..dst_offset + len_w]
                        .copy_from_slice(&data[src_offset..src_offset + len_w]);
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
            let mut output = vec![0.0f64; batch_size * len_h * len_w];

            for b in 0..batch_size {
                for row in 0..len_h {
                    let src_row = start_h + row;
                    let src_offset = b * src_h * src_w + src_row * src_w + start_w;
                    let dst_offset = b * len_h * len_w + row * len_w;
                    output[dst_offset..dst_offset + len_w]
                        .copy_from_slice(&data[src_offset..src_offset + len_w]);
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
            op: "slice_last_2d",
        }),
    }
}
