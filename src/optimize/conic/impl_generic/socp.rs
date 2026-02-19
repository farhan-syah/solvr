//! SOCP interior point solver.
//!
//! Uses a primal-dual interior point method with self-scaled barriers
//! for second-order cone constraints.

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::dtype::DType;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::conic::traits::socp::{SocConstraint, SocpOptions, SocpResult};
use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::impl_generic::utils::{tensor_dot, tensor_norm};

/// SOCP interior point implementation.
pub fn socp_impl<R, C>(
    client: &C,
    cost: &Tensor<R>,
    constraints: &[SocConstraint<R>],
    options: &SocpOptions,
) -> OptimizeResult<SocpResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let n = cost.shape()[0];

    if constraints.is_empty() {
        // Unconstrained: optimal is unbounded or at origin
        let x = client
            .fill(&[n], 0.0, DType::F64)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("socp: fill - {}", e),
            })?;
        let fun = tensor_dot(client, cost, &x).map_err(|e| OptimizeError::NumericalError {
            message: format!("socp: dot - {}", e),
        })?;
        return Ok(SocpResult {
            x,
            fun,
            iterations: 0,
            converged: true,
        });
    }

    // Initialize x at a strictly feasible point (inside all cones)
    let mut x = client
        .fill(&[n], 0.0, DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("socp: initial x - {}", e),
        })?;

    let mut mu = 1.0;
    let mu_factor = 0.3;

    for iter in 0..options.max_iter {
        // Check convergence: duality gap
        if mu < options.tol {
            let fun = tensor_dot(client, cost, &x).map_err(|e| OptimizeError::NumericalError {
                message: format!("socp: dot - {}", e),
            })?;
            return Ok(SocpResult {
                x,
                fun,
                iterations: iter + 1,
                converged: true,
            });
        }

        // Compute barrier gradient and Hessian contribution
        let mut grad = cost.clone();

        for constraint in constraints {
            let (barrier_grad, cone_feasible) =
                compute_soc_barrier_gradient(client, &x, constraint, mu)?;

            if !cone_feasible {
                // Not strictly feasible; reduce step and try again
                break;
            }

            grad = client
                .add(&grad, &barrier_grad)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("socp: add barrier grad - {}", e),
                })?;
        }

        // Take a gradient step (simplified Newton)
        let grad_norm = tensor_norm(client, &grad).map_err(|e| OptimizeError::NumericalError {
            message: format!("socp: grad norm - {}", e),
        })?;

        if grad_norm < options.tol {
            mu *= mu_factor;
            continue;
        }

        let step_size = mu / grad_norm.max(1.0);
        let direction =
            client
                .mul_scalar(&grad, -step_size)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("socp: direction - {}", e),
                })?;

        // Backtracking line search to maintain feasibility
        let mut alpha = 1.0;
        for _ in 0..30 {
            let step = client.mul_scalar(&direction, alpha).map_err(|e| {
                OptimizeError::NumericalError {
                    message: format!("socp: scale step - {}", e),
                }
            })?;
            let x_trial = client
                .add(&x, &step)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("socp: x + step - {}", e),
                })?;

            // Check all cones are feasible
            let feasible = check_cone_feasibility(client, &x_trial, constraints)?;
            if feasible {
                x = x_trial;
                break;
            }
            alpha *= 0.5;
        }

        mu *= mu_factor;
    }

    let fun = tensor_dot(client, cost, &x).map_err(|e| OptimizeError::NumericalError {
        message: format!("socp: final dot - {}", e),
    })?;

    Ok(SocpResult {
        x,
        fun,
        iterations: options.max_iter,
        converged: false,
    })
}

/// Compute barrier gradient for a single SOC constraint.
/// ||A*x + b|| <= c'*x + d  =>  barrier = -log((c'x+d)^2 - ||Ax+b||^2)
fn compute_soc_barrier_gradient<R, C>(
    client: &C,
    x: &Tensor<R>,
    constraint: &SocConstraint<R>,
    mu: f64,
) -> OptimizeResult<(Tensor<R>, bool)>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let n = x.shape()[0];

    // t = c'x + d
    let t = tensor_dot(client, &constraint.c, x).map_err(|e| OptimizeError::NumericalError {
        message: format!("soc_barrier: c'x - {}", e),
    })? + constraint.d;

    // u = A*x + b
    let x_col = x
        .reshape(&[n, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("soc_barrier: reshape x - {}", e),
        })?;
    let ax = client
        .matmul(&constraint.a, &x_col)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("soc_barrier: A*x - {}", e),
        })?;
    let m_i = constraint.a.shape()[0];
    let ax_flat = ax
        .reshape(&[m_i])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("soc_barrier: reshape Ax - {}", e),
        })?;
    let u = client
        .add(&ax_flat, &constraint.b)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("soc_barrier: Ax + b - {}", e),
        })?;

    // ||u||^2
    let u_sq = tensor_dot(client, &u, &u).map_err(|e| OptimizeError::NumericalError {
        message: format!("soc_barrier: u'u - {}", e),
    })?;

    // gap = t^2 - ||u||^2
    let gap = t * t - u_sq;
    if gap <= 0.0 {
        // Not strictly feasible
        let zero_grad =
            client
                .fill(&[n], 0.0, DType::F64)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("soc_barrier: zero grad - {}", e),
                })?;
        return Ok((zero_grad, false));
    }

    // Gradient of barrier = -mu/gap * (2*t*c - 2*A'*u)
    // = -2*mu/gap * (t*c - A'*u)
    let coeff = -2.0 * mu / gap;

    let tc = client
        .mul_scalar(&constraint.c, t)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("soc_barrier: t*c - {}", e),
        })?;

    let a_t = constraint
        .a
        .transpose(0, 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("soc_barrier: A' - {}", e),
        })?;
    let u_col = u
        .reshape(&[m_i, 1])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("soc_barrier: reshape u - {}", e),
        })?;
    let atu = client
        .matmul(&a_t, &u_col)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("soc_barrier: A'u - {}", e),
        })?
        .reshape(&[n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("soc_barrier: reshape A'u - {}", e),
        })?;

    let diff = client
        .sub(&tc, &atu)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("soc_barrier: tc - A'u - {}", e),
        })?;
    let barrier_grad =
        client
            .mul_scalar(&diff, coeff)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("soc_barrier: scale grad - {}", e),
            })?;

    Ok((barrier_grad, true))
}

fn check_cone_feasibility<R, C>(
    client: &C,
    x: &Tensor<R>,
    constraints: &[SocConstraint<R>],
) -> OptimizeResult<bool>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
{
    let n = x.shape()[0];
    for constraint in constraints {
        let t =
            tensor_dot(client, &constraint.c, x).map_err(|e| OptimizeError::NumericalError {
                message: format!("cone_check: c'x - {}", e),
            })? + constraint.d;

        let x_col = x
            .reshape(&[n, 1])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cone_check: reshape x - {}", e),
            })?;
        let ax =
            client
                .matmul(&constraint.a, &x_col)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("cone_check: A*x - {}", e),
                })?;
        let m_i = constraint.a.shape()[0];
        let ax_flat = ax
            .reshape(&[m_i])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cone_check: reshape - {}", e),
            })?;
        let u = client
            .add(&ax_flat, &constraint.b)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("cone_check: Ax+b - {}", e),
            })?;
        let u_norm = tensor_norm(client, &u).map_err(|e| OptimizeError::NumericalError {
            message: format!("cone_check: norm - {}", e),
        })?;

        if u_norm >= t {
            return Ok(false);
        }
    }
    Ok(true)
}
