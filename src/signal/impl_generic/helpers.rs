//! Generic helper functions for signal processing.
//!
//! These functions implement tensor operations not available in TensorOps,
//! using to_vec()/from_slice() as a universal fallback for any Runtime.

use numr::dtype::{Complex64, Complex128, DType};
use numr::error::{Error, Result};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Reverse 1D tensor (generic over Runtime).
pub fn reverse_1d_impl<R, C>(client: &C, tensor: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let dtype = tensor.dtype();
    let tensor_contig = tensor.contiguous();

    if tensor_contig.ndim() != 1 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "reverse_1d requires 1D tensor".to_string(),
        });
    }

    let len = tensor_contig.shape()[0];

    match dtype {
        DType::F32 => {
            let data: Vec<f32> = tensor_contig.to_vec();
            let reversed: Vec<f32> = data.into_iter().rev().collect();
            Ok(Tensor::<R>::from_slice(&reversed, &[len], client.device()))
        }
        DType::F64 => {
            let data: Vec<f64> = tensor_contig.to_vec();
            let reversed: Vec<f64> = data.into_iter().rev().collect();
            Ok(Tensor::<R>::from_slice(&reversed, &[len], client.device()))
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "reverse_1d",
        }),
    }
}

/// Reverse 2D tensor - flip both dimensions (generic over Runtime).
pub fn reverse_2d_impl<R, C>(client: &C, tensor: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let dtype = tensor.dtype();
    let tensor_contig = tensor.contiguous();

    if tensor_contig.ndim() != 2 {
        return Err(Error::InvalidArgument {
            arg: "tensor",
            reason: "reverse_2d requires 2D tensor".to_string(),
        });
    }

    let h = tensor_contig.shape()[0];
    let w = tensor_contig.shape()[1];

    match dtype {
        DType::F32 => {
            let data: Vec<f32> = tensor_contig.to_vec();
            let mut reversed = vec![0.0f32; h * w];
            for i in 0..h {
                for j in 0..w {
                    reversed[i * w + j] = data[(h - 1 - i) * w + (w - 1 - j)];
                }
            }
            Ok(Tensor::<R>::from_slice(&reversed, &[h, w], client.device()))
        }
        DType::F64 => {
            let data: Vec<f64> = tensor_contig.to_vec();
            let mut reversed = vec![0.0f64; h * w];
            for i in 0..h {
                for j in 0..w {
                    reversed[i * w + j] = data[(h - 1 - i) * w + (w - 1 - j)];
                }
            }
            Ok(Tensor::<R>::from_slice(&reversed, &[h, w], client.device()))
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "reverse_2d",
        }),
    }
}

/// Element-wise complex multiplication (generic over Runtime).
pub fn complex_mul_impl<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let dtype = a.dtype();

    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    if a.shape() != b.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    let a_contig = a.contiguous();
    let b_contig = b.contiguous();

    match dtype {
        DType::Complex64 => {
            let a_data: Vec<Complex64> = a_contig.to_vec();
            let b_data: Vec<Complex64> = b_contig.to_vec();
            let result: Vec<Complex64> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(av, bv)| {
                    Complex64::new(av.re * bv.re - av.im * bv.im, av.re * bv.im + av.im * bv.re)
                })
                .collect();
            Ok(Tensor::<R>::from_slice(&result, a.shape(), client.device()))
        }
        DType::Complex128 => {
            let a_data: Vec<Complex128> = a_contig.to_vec();
            let b_data: Vec<Complex128> = b_contig.to_vec();
            let result: Vec<Complex128> = a_data
                .iter()
                .zip(b_data.iter())
                .map(|(av, bv)| {
                    Complex128::new(av.re * bv.re - av.im * bv.im, av.re * bv.im + av.im * bv.re)
                })
                .collect();
            Ok(Tensor::<R>::from_slice(&result, a.shape(), client.device()))
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "complex_mul",
        }),
    }
}

/// Compute |complex|^power for spectrogram (generic over Runtime).
pub fn complex_magnitude_pow_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    power: f64,
    output_dtype: DType,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let dtype = tensor.dtype();
    let tensor_contig = tensor.contiguous();

    match (dtype, output_dtype) {
        (DType::Complex64, DType::F32) => {
            let data: Vec<Complex64> = tensor_contig.to_vec();
            let result: Vec<f32> = data
                .iter()
                .map(|c| {
                    let mag_sq = c.re * c.re + c.im * c.im;
                    if power == 2.0 {
                        mag_sq
                    } else if power == 1.0 {
                        mag_sq.sqrt()
                    } else {
                        mag_sq.powf(power as f32 / 2.0)
                    }
                })
                .collect();
            Ok(Tensor::<R>::from_slice(
                &result,
                tensor.shape(),
                client.device(),
            ))
        }
        (DType::Complex128, DType::F64) => {
            let data: Vec<Complex128> = tensor_contig.to_vec();
            let result: Vec<f64> = data
                .iter()
                .map(|c| {
                    let mag_sq = c.re * c.re + c.im * c.im;
                    if power == 2.0 {
                        mag_sq
                    } else if power == 1.0 {
                        mag_sq.sqrt()
                    } else {
                        mag_sq.powf(power / 2.0)
                    }
                })
                .collect();
            Ok(Tensor::<R>::from_slice(
                &result,
                tensor.shape(),
                client.device(),
            ))
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "complex_magnitude_pow",
        }),
    }
}
