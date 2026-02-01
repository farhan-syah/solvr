//! Utility functions for optimization algorithms using tensors.

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute the L2 norm of a 1D tensor.
pub fn tensor_norm<R, C>(client: &C, x: &Tensor<R>) -> Result<f64>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let x_sq = client.mul(x, x)?;
    let sum = client.sum(&x_sq, &[0], false)?;
    let sum_val: Vec<f64> = sum.to_vec();
    Ok(sum_val[0].sqrt())
}

/// Compute dot product of two 1D tensors.
pub fn tensor_dot<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<f64>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let prod = client.mul(a, b)?;
    let sum = client.sum(&prod, &[0], false)?;
    let sum_val: Vec<f64> = sum.to_vec();
    Ok(sum_val[0])
}

/// Compute forward finite difference gradient using tensor operations.
///
/// For each dimension i, computes (f(x + eps*e_i) - f(x)) / eps
/// where e_i is the i-th unit vector.
///
/// Note: This is inherently O(n) function evaluations. We keep x on device
/// and only create small delta tensors for each perturbation direction.
pub fn finite_difference_gradient<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    fx: f64,
    eps: f64,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let n = x.shape()[0];
    let mut grad_vals = vec![0.0; n];

    for i in 0..n {
        // Create delta = eps * e_i (unit vector scaled by eps)
        let mut delta_data = vec![0.0; n];
        delta_data[i] = eps;
        let delta = Tensor::<R>::from_slice(&delta_data, &[n], client.device());

        // x_plus = x + delta (tensor addition on device)
        let x_plus = client.add(x, &delta)?;
        let fx_plus = f(&x_plus)?;
        grad_vals[i] = (fx_plus - fx) / eps;
    }

    Ok(Tensor::<R>::from_slice(&grad_vals, &[n], client.device()))
}

/// Add two tensors element-wise.
pub fn tensor_add<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    client.add(a, b)
}

/// Subtract two tensors element-wise.
pub fn tensor_sub<R, C>(client: &C, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    client.sub(a, b)
}

/// Scale tensor by a scalar.
pub fn tensor_scale<R, C>(client: &C, x: &Tensor<R>, s: f64) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + RuntimeClient<R>,
{
    client.mul_scalar(x, s)
}

/// Threshold for detecting singular/near-zero values.
pub const SINGULAR_THRESHOLD: f64 = 1e-12;
