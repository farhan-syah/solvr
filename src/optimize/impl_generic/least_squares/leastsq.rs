//! Levenberg-Marquardt algorithm for unbounded nonlinear least squares using tensors.

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::impl_generic::utils::SINGULAR_THRESHOLD;
use crate::optimize::least_squares::LeastSquaresOptions;

use super::TensorLeastSquaresResult;

/// Levenberg-Marquardt algorithm for nonlinear least squares using tensors.
///
/// Minimizes ||f(x)||^2 where f: R^n -> R^m.
pub fn leastsq_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    options: &LeastSquaresOptions,
) -> OptimizeResult<TensorLeastSquaresResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    let n = x0.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "leastsq: empty initial guess".to_string(),
        });
    }

    let mut x = x0.clone();
    let mut fx = f(&x).map_err(|e| OptimizeError::NumericalError {
        message: format!("leastsq: initial evaluation - {}", e),
    })?;

    let m = fx.shape()[0];
    if m == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "leastsq: residual function returns empty vector".to_string(),
        });
    }

    let mut nfev = 1;
    let mut cost = compute_cost(client, &fx)?;

    let mut lambda = 0.001;
    let lambda_up = 10.0;
    let lambda_down = 0.1;
    let lambda_min = SINGULAR_THRESHOLD;
    let lambda_max = 1e10;

    for iter in 0..options.max_iter {
        if cost < options.f_tol {
            return Ok(TensorLeastSquaresResult {
                x,
                residuals: fx,
                cost,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute Jacobian using finite differences
        let jacobian = finite_difference_jacobian(client, &f, &x, &fx, m, n, options.eps)?;
        nfev += n;

        // Compute J^T J using numr's transpose
        let jt = jacobian
            .transpose(0, 1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("leastsq: transpose - {}", e),
            })?;
        let jtj = client
            .matmul(&jt, &jacobian)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("leastsq: J^T J - {}", e),
            })?;

        // Add lambda * diag(J^T J) damping
        let jtj_damped = add_scaled_diagonal(client, &jtj, lambda, n)?;

        // Compute J^T f
        let fx_col = fx
            .reshape(&[m, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("leastsq: reshape fx - {}", e),
            })?;
        let jtf = client
            .matmul(&jt, &fx_col)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("leastsq: J^T f - {}", e),
            })?;

        // Check gradient norm for convergence
        let jtf_vec = jtf
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("leastsq: reshape jtf - {}", e),
            })?;
        let grad_norm = tensor_norm(client, &jtf_vec)?;
        if grad_norm < options.g_tol {
            return Ok(TensorLeastSquaresResult {
                x,
                residuals: fx,
                cost,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Solve (J^T J + lambda*diag) * dx = -J^T f using numr's solve
        let neg_jtf = client
            .mul_scalar(&jtf, -1.0)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("leastsq: negate jtf - {}", e),
            })?;

        let dx_col = match TensorOps::solve(client, &jtj_damped, &neg_jtf) {
            Ok(dx) => dx,
            Err(_) => {
                lambda *= lambda_up;
                lambda = lambda.clamp(lambda_min, lambda_max);
                continue;
            }
        };

        let dx = dx_col
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("leastsq: reshape dx - {}", e),
            })?;

        // x_new = x + dx
        let x_new = client
            .add(&x, &dx)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("leastsq: update x - {}", e),
            })?;

        let fx_new = f(&x_new).map_err(|e| OptimizeError::NumericalError {
            message: format!("leastsq: evaluation - {}", e),
        })?;
        nfev += 1;

        let cost_new = compute_cost(client, &fx_new)?;

        if cost_new < cost {
            // Accept step
            let dx_norm = tensor_norm(client, &dx)?;
            if dx_norm < options.x_tol {
                return Ok(TensorLeastSquaresResult {
                    x: x_new,
                    residuals: fx_new,
                    cost: cost_new,
                    iterations: iter + 1,
                    nfev,
                    converged: true,
                });
            }

            x = x_new;
            fx = fx_new;
            cost = cost_new;
            lambda *= lambda_down;
        } else {
            // Reject step, increase damping
            lambda *= lambda_up;
        }

        lambda = lambda.clamp(lambda_min, lambda_max);
    }

    Ok(TensorLeastSquaresResult {
        x,
        residuals: fx,
        cost,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

/// Compute cost = ||f||^2 = sum(f_i^2)
fn compute_cost<R, C>(client: &C, fx: &Tensor<R>) -> OptimizeResult<f64>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let fx_sq = client
        .mul(fx, fx)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("compute_cost: mul - {}", e),
        })?;
    let sum = client
        .sum(&fx_sq, &[0], false)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("compute_cost: sum - {}", e),
        })?;
    let vals: Vec<f64> = sum.to_vec();
    Ok(vals[0])
}

/// Compute L2 norm of a vector tensor.
fn tensor_norm<R, C>(client: &C, v: &Tensor<R>) -> OptimizeResult<f64>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let cost = compute_cost(client, v)?;
    Ok(cost.sqrt())
}

/// Compute Jacobian matrix using forward finite differences.
/// Returns [m, n] tensor where J[i,j] = df_i/dx_j.
///
/// All operations stay on device - no to_vec()/from_slice().
fn finite_difference_jacobian<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    fx: &Tensor<R>,
    _m: usize,
    n: usize,
    eps: f64,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    // Create identity matrix [n, n] scaled by eps
    let identity = client
        .eye(n, None, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("jacobian: eye - {}", e),
        })?;
    let eps_identity = client
        .mul_scalar(&identity, eps)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("jacobian: scale identity - {}", e),
        })?;

    // Compute each column of the Jacobian
    let mut jac_columns: Vec<Tensor<R>> = Vec::with_capacity(n);

    for j in 0..n {
        // Extract row j as delta vector (identity is symmetric, so row = column)
        let delta = eps_identity
            .narrow(0, j, 1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("jacobian: narrow row - {}", e),
            })?
            .contiguous()
            .reshape(&[n])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("jacobian: reshape delta - {}", e),
            })?;

        // x_plus = x + delta
        let x_plus = client
            .add(x, &delta)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("jacobian: x + delta - {}", e),
            })?;

        // f(x_plus)
        let fx_plus = f(&x_plus).map_err(|e| OptimizeError::NumericalError {
            message: format!("jacobian: f(x+delta) - {}", e),
        })?;

        // jac_col = (fx_plus - fx) / eps, shape [m]
        let diff = client
            .sub(&fx_plus, fx)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("jacobian: fx_plus - fx - {}", e),
            })?;
        let jac_col = client
            .mul_scalar(&diff, 1.0 / eps)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("jacobian: scale diff - {}", e),
            })?;

        // Reshape to [m, 1] for concatenation
        let jac_col_2d = jac_col
            .unsqueeze(1)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("jacobian: unsqueeze col - {}", e),
            })?;
        jac_columns.push(jac_col_2d);
    }

    // Concatenate columns: [m, 1] * n -> [m, n]
    let refs: Vec<&Tensor<R>> = jac_columns.iter().collect();
    client
        .cat(&refs, 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("jacobian: cat columns - {}", e),
        })
}

/// Add lambda * max(|diag(A)|, threshold) to diagonal of A.
/// Uses tensor ops throughout - no to_vec()/from_slice().
fn add_scaled_diagonal<R, C>(
    client: &C,
    a: &Tensor<R>,
    lambda: f64,
    n: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    // Extract diagonal using numr's diag
    let diag_vec = TensorOps::diag(client, a)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_scaled_diagonal: diag - {}", e),
        })?;

    // Compute abs(diag)
    let abs_diag = client
        .abs(&diag_vec)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_scaled_diagonal: abs - {}", e),
        })?;

    // Create threshold tensor
    let threshold = client
        .fill(&[n], SINGULAR_THRESHOLD, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_scaled_diagonal: threshold - {}", e),
        })?;

    // max(|diag|, threshold)
    let clamped_diag = client
        .maximum(&abs_diag, &threshold)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_scaled_diagonal: max - {}", e),
        })?;

    // Scale by lambda
    let scaled_diag = client
        .mul_scalar(&clamped_diag, lambda)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_scaled_diagonal: scale - {}", e),
        })?;

    // Create diagonal matrix using numr's diagflat
    let diag_matrix = TensorOps::diagflat(client, &scaled_diag)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_scaled_diagonal: diagflat - {}", e),
        })?;

    client
        .add(a, &diag_matrix)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("add_scaled_diagonal: add - {}", e),
        })
}
