//! Utility functions for optimization algorithms using tensors.

use numr::dtype::DType;
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
/// All operations stay on device - no to_vec()/from_slice().
pub fn finite_difference_gradient<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    fx: f64,
    eps: f64,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let n = x.shape()[0];

    // Create identity matrix scaled by eps - each row is eps * e_i
    let identity = client.eye(n, None, DType::F64)?;
    let eps_identity = client.mul_scalar(&identity, eps)?;

    // Compute gradient components
    let mut grad_components: Vec<Tensor<R>> = Vec::with_capacity(n);

    for i in 0..n {
        // Extract row i as delta vector using narrow
        let delta = eps_identity
            .narrow(0, i, 1)?
            .contiguous()
            .reshape(&[n])?;

        // x_plus = x + delta (tensor addition on device)
        let x_plus = client.add(x, &delta)?;
        let fx_plus = f(&x_plus)?;

        // Create gradient component as 1-element tensor
        let grad_i = (fx_plus - fx) / eps;
        let grad_i_tensor = client.fill(&[1], grad_i, DType::F64)?;
        grad_components.push(grad_i_tensor);
    }

    // Concatenate all gradient components into final gradient vector
    let refs: Vec<&Tensor<R>> = grad_components.iter().collect();
    client.cat(&refs, 0)
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
