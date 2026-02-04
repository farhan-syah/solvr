//! Autograd-based Jacobian computation shared across solvr.
//!
//! Provides exact Jacobian computation using numr's forward-mode automatic
//! differentiation. This is used by:
//! - `optimize/roots` - Newton's method for root finding
//! - `integrate/ode` - Implicit ODE solvers (BDF, Radau, LSODA)
//!
//! # Why Autograd over Finite Differences?
//!
//! Finite differences have several drawbacks:
//! - Truncation error from finite step size
//! - Numerical instability for ill-conditioned problems
//! - Requires tuning epsilon parameter
//! - O(n) function evaluations per Jacobian
//!
//! Autograd provides:
//! - Exact derivatives (to machine precision)
//! - No epsilon tuning required
//! - More robust for stiff problems
//! - Same O(n) cost but exact results
//!
//! # Usage
//!
//! User must write their function using `DualTensor` and `dual_*` operations:
//!
//! ```ignore
//! use numr::autograd::dual_ops::{dual_mul, dual_sub};
//!
//! // f(y) = [y[0]^2 - y[1], y[0] * y[1] - 1]
//! let jacobian = jacobian_autograd(&client, |y, c| {
//!     // Use dual operations for automatic differentiation
//!     let y0_sq = dual_mul(&y[0], &y[0], c)?;
//!     // ... build output using dual ops
//! }, &y)?;
//! ```

use numr::autograd::{DualTensor, jacobian_forward, jvp};
use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Compute the Jacobian matrix using forward-mode automatic differentiation.
///
/// For a function F: ℝⁿ → ℝᵐ, computes the m×n Jacobian matrix J where
/// J[i,j] = ∂Fᵢ/∂xⱼ.
///
/// This uses numr's forward-mode AD, which computes n JVPs (one per input
/// dimension). For square systems (n = m), this is optimal.
///
/// # Arguments
///
/// * `client` - Runtime client for tensor operations
/// * `f` - Function using `DualTensor` and `dual_*` operations
/// * `x` - Point at which to evaluate the Jacobian
///
/// # Returns
///
/// Jacobian matrix of shape [m, n]
///
/// # Example
///
/// ```ignore
/// use numr::autograd::dual_ops::dual_mul;
///
/// // F(x) = x² (element-wise), Jacobian = diag(2x)
/// let jacobian = jacobian_autograd(
///     &client,
///     |dual_x, c| dual_mul(dual_x, dual_x, c),
///     &x,
/// )?;
/// ```
pub fn jacobian_autograd<R, C, F>(client: &C, f: F, x: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&DualTensor<R>, &C) -> Result<DualTensor<R>>,
{
    jacobian_forward(f, x, client)
}

/// Compute Jacobian-vector product J @ v using forward-mode AD.
///
/// For a function F: ℝⁿ → ℝᵐ, computes J(x) @ v without forming the full
/// Jacobian matrix. This is useful for:
/// - Newton-Krylov methods (iterative linear solvers)
/// - Large systems where forming J is expensive
/// - Memory-constrained environments
///
/// # Arguments
///
/// * `client` - Runtime client
/// * `f` - Function using `DualTensor` operations
/// * `x` - Point at which to evaluate
/// * `v` - Vector to multiply with Jacobian
///
/// # Returns
///
/// Tuple of (F(x), J(x) @ v)
pub fn jvp_autograd<R, C, F>(
    client: &C,
    f: F,
    x: &Tensor<R>,
    v: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: FnOnce(&[DualTensor<R>], &C) -> Result<DualTensor<R>>,
{
    jvp(f, &[x], &[v], client)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::autograd::dual_ops::{dual_mul, dual_mul_scalar};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_jacobian_autograd_linear() {
        let (device, client) = setup();

        // F(x) = 2x, Jacobian = 2I
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);

        let jacobian =
            jacobian_autograd(&client, |dual_x, c| dual_mul_scalar(dual_x, 2.0, c), &x).unwrap();

        assert_eq!(jacobian.shape(), &[3, 3]);
        let j: Vec<f64> = jacobian.to_vec();

        // Diagonal should be 2, off-diagonal 0
        assert!((j[0] - 2.0).abs() < 1e-10);
        assert!((j[4] - 2.0).abs() < 1e-10);
        assert!((j[8] - 2.0).abs() < 1e-10);
        assert!(j[1].abs() < 1e-10);
        assert!(j[2].abs() < 1e-10);
        assert!(j[3].abs() < 1e-10);
    }

    #[test]
    fn test_jacobian_autograd_quadratic() {
        let (device, client) = setup();

        // F(x) = x², Jacobian = diag(2x)
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);

        let jacobian =
            jacobian_autograd(&client, |dual_x, c| dual_mul(dual_x, dual_x, c), &x).unwrap();

        let j: Vec<f64> = jacobian.to_vec();

        // Diagonal should be [2, 4, 6]
        assert!((j[0] - 2.0).abs() < 1e-10);
        assert!((j[4] - 4.0).abs() < 1e-10);
        assert!((j[8] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_jvp_autograd() {
        let (device, client) = setup();

        // F(x) = x², at x=[2], v=[1]
        // F(x) = 4, J @ v = 2*2*1 = 4
        let x = Tensor::<CpuRuntime>::from_slice(&[2.0f64], &[1], &device);
        let v = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);

        let (fx, jv) = jvp_autograd(
            &client,
            |inputs, c| {
                let x = &inputs[0];
                dual_mul(x, x, c)
            },
            &x,
            &v,
        )
        .unwrap();

        assert!((fx.to_vec::<f64>()[0] - 4.0).abs() < 1e-10);
        assert!((jv.to_vec::<f64>()[0] - 4.0).abs() < 1e-10);
    }
}
