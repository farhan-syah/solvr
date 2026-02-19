//! Generic trust region outer loop, parameterized by subproblem solver.
//!
//! The trust region method iteratively:
//! 1. Solves a subproblem: min m(p) = f + g^T p + 0.5 p^T H p  s.t. ||p|| <= delta
//! 2. Evaluates actual reduction vs predicted reduction
//! 3. Updates trust region radius based on the ratio
//!
//! Different subproblem solvers (NCG, exact, Krylov) plug into this framework.
use crate::DType;

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::traits::trust_region::{TrustRegionOptions, TrustRegionResult};

use super::helpers::{gradient_from_fn, hvp_from_fn};
use super::utils::{tensor_dot, tensor_norm};

/// Result from a trust region subproblem solver.
pub struct SubproblemResult<R: Runtime<DType = DType>> {
    /// The step p to take
    pub step: Tensor<R>,
    /// Whether the step hits the trust region boundary
    pub hits_boundary: bool,
    /// Predicted reduction: -m(p) = -(g^T p + 0.5 p^T H p)
    pub predicted_reduction: f64,
}

/// Trait for trust region subproblem solvers.
pub trait SubproblemSolver<R: Runtime<DType = DType>, C, F> {
    /// Solve the trust region subproblem.
    ///
    /// Find p that approximately minimizes g^T p + 0.5 p^T H p subject to ||p|| <= delta.
    ///
    /// # Arguments
    /// * `client` - Runtime client
    /// * `f` - The objective function (for HVP computation)
    /// * `x` - Current point (for HVP computation)
    /// * `grad` - Gradient at current point
    /// * `trust_radius` - Current trust region radius
    fn solve(
        &self,
        client: &C,
        f: &F,
        x: &Tensor<R>,
        grad: &Tensor<R>,
        trust_radius: f64,
    ) -> OptimizeResult<SubproblemResult<R>>;
}

/// Generic trust region outer loop.
///
/// This function implements the trust region framework. The subproblem solver S
/// determines which variant (NCG, exact, Krylov) is used.
pub fn trust_region_loop<R, C, F, S>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    options: &TrustRegionOptions,
    solver: &S,
) -> OptimizeResult<TrustRegionResult<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
    F: Fn(&Var<R>, &C) -> NumrResult<Var<R>>,
    S: SubproblemSolver<R, C, F>,
{
    if x0.numel() == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "trust_region: empty initial guess".to_string(),
        });
    }

    // Validate options
    if options.initial_trust_radius <= 0.0 {
        return Err(OptimizeError::InvalidInput {
            context: "trust_region: initial_trust_radius must be positive".to_string(),
        });
    }
    if options.max_trust_radius <= options.initial_trust_radius {
        return Err(OptimizeError::InvalidInput {
            context: "trust_region: max_trust_radius must exceed initial_trust_radius".to_string(),
        });
    }
    if options.eta < 0.0 || options.eta >= 0.25 {
        return Err(OptimizeError::InvalidInput {
            context: "trust_region: eta must be in [0, 0.25)".to_string(),
        });
    }

    let mut x = x0.clone();
    let mut delta = options.initial_trust_radius;
    let mut nfev = 0;

    // Initial evaluation
    let (mut fx, mut grad) = gradient_from_fn(client, &f, &x)?;
    nfev += 1;

    for iter in 0..options.max_iter {
        // Check gradient convergence
        let grad_norm = tensor_norm(client, &grad).map_err(|e| OptimizeError::NumericalError {
            message: format!("trust_region: grad norm - {}", e),
        })?;

        if grad_norm < options.gtol {
            return Ok(TrustRegionResult {
                x,
                fun: fx,
                grad,
                iterations: iter,
                converged: true,
                trust_radius: delta,
                nfev,
            });
        }

        // Solve subproblem
        let sub_result = solver.solve(client, &f, &x, &grad, delta)?;

        // Compute actual reduction
        let step = &sub_result.step;
        let x_new = client
            .add(&x, step)
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("trust_region: x + step - {}", e),
            })?;

        let (fx_new, grad_new) = gradient_from_fn(client, &f, &x_new)?;
        nfev += 1;

        let actual_reduction = fx - fx_new;
        let predicted_reduction = sub_result.predicted_reduction;

        // If predicted reduction is not positive, the subproblem failed
        // to find a descent step - shrink trust region and try again
        if predicted_reduction <= 0.0 {
            delta *= 0.25;
            if delta < 1e-15 {
                return Ok(TrustRegionResult {
                    x,
                    fun: fx,
                    grad,
                    iterations: iter + 1,
                    converged: false,
                    trust_radius: delta,
                    nfev,
                });
            }
            continue;
        }

        // Compute ratio of actual to predicted reduction
        let rho = if predicted_reduction.abs() < 1e-30 {
            // If predicted reduction is essentially zero, treat as successful
            // if actual reduction is also essentially zero
            if actual_reduction.abs() < 1e-30 {
                1.0
            } else {
                0.0
            }
        } else {
            actual_reduction / predicted_reduction
        };

        // Update trust region radius
        if rho < 0.25 {
            // Poor agreement: shrink trust region
            delta *= 0.25;
        } else if rho > 0.75 && sub_result.hits_boundary {
            // Good agreement and we hit the boundary: expand
            delta = (2.0 * delta).min(options.max_trust_radius);
        }
        // else: keep delta unchanged

        // Accept or reject step
        if rho > options.eta {
            // Accept step
            x = x_new;
            fx = fx_new;
            grad = grad_new;
        }

        // Check if trust region is too small
        if delta < 1e-15 {
            return Ok(TrustRegionResult {
                x,
                fun: fx,
                grad,
                iterations: iter + 1,
                converged: false,
                trust_radius: delta,
                nfev,
            });
        }
    }

    // Did not converge
    Ok(TrustRegionResult {
        x,
        fun: fx,
        grad,
        iterations: options.max_iter,
        converged: false,
        trust_radius: delta,
        nfev,
    })
}

/// Compute predicted reduction for step p given gradient g and HVP Hp.
///
/// predicted_reduction = -(g^T p + 0.5 p^T H p)
pub fn compute_predicted_reduction<R, C>(
    client: &C,
    grad: &Tensor<R>,
    step: &Tensor<R>,
    h_step: &Tensor<R>,
) -> OptimizeResult<f64>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    // g^T p
    let g_dot_p = tensor_dot(client, grad, step).map_err(|e| OptimizeError::NumericalError {
        message: format!("trust_region: dot - {}", e),
    })?;

    // p^T H p
    let p_dot_hp = tensor_dot(client, step, h_step).map_err(|e| OptimizeError::NumericalError {
        message: format!("trust_region: dot - {}", e),
    })?;

    // predicted_reduction = -(g^T p + 0.5 * p^T H p)
    Ok(-(g_dot_p + 0.5 * p_dot_hp))
}

/// Safeguarded Newton update on the secular equation phi(lambda) = ||p|| - delta.
///
/// Given the current lambda and the ratio ||p|| / delta, compute the next lambda
/// that drives ||p|| toward delta.
pub fn secular_newton_update(lambda: f64, ratio: f64) -> f64 {
    let updated = lambda + (ratio - 1.0) * lambda / ratio;
    updated.max(1e-15)
}

/// Compute Hessian-vector product using autograd (wrapper for subproblem solvers).
pub fn compute_hvp_for_subproblem<R, C, F>(
    client: &C,
    f: &F,
    x: &Tensor<R>,
    v: &Tensor<R>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
    F: Fn(&Var<R>, &C) -> NumrResult<Var<R>>,
{
    let (_fx, hvp) = hvp_from_fn(client, f, x, v)?;
    Ok(hvp)
}
