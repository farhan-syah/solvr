//! Utility functions for optimization algorithms using tensors.

use numr::error::Result;
use numr::ops::TensorOps;
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

/// Compute forward finite difference gradient.
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
    let x_data: Vec<f64> = x.to_vec();
    let mut grad = vec![0.0; n];

    for i in 0..n {
        let mut x_plus = x_data.clone();
        x_plus[i] += eps;
        let x_plus_tensor = Tensor::<R>::from_slice(&x_plus, &[n], client.device());
        let fx_plus = f(&x_plus_tensor)?;
        grad[i] = (fx_plus - fx) / eps;
    }

    Ok(Tensor::<R>::from_slice(&grad, &[n], client.device()))
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
    C: TensorOps<R> + RuntimeClient<R>,
{
    let n = x.shape()[0];
    let scale_tensor = Tensor::<R>::from_slice(&vec![s; n], &[n], client.device());
    client.mul(x, &scale_tensor)
}

/// Threshold for detecting singular/near-zero values.
pub const SINGULAR_THRESHOLD: f64 = 1e-12;
