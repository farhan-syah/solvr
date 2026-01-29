//! Multivariate root finding algorithms.
//!
//! This module provides methods for finding roots of systems of nonlinear equations.
//! Given F: R^n -> R^n, find x such that F(x) = 0.

// Indexed loops are clearer for matrix operations
#![allow(clippy::needless_range_loop)]

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::utils::{
    SINGULAR_THRESHOLD, ZERO_THRESHOLD, finite_difference_jacobian, norm, solve_linear_system,
};

/// Options for multivariate root finding.
#[derive(Debug, Clone)]
pub struct RootOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence (norm of F(x))
    pub tol: f64,
    /// Tolerance for step size
    pub x_tol: f64,
    /// Step size for finite difference Jacobian approximation
    pub eps: f64,
}

impl Default for RootOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-8,
            x_tol: 1e-8,
            eps: 1e-8,
        }
    }
}

/// Result from a multivariate root finding method.
#[derive(Debug, Clone)]
pub struct MultiRootResult {
    /// The root found
    pub x: Vec<f64>,
    /// Function value at root (should be near zero)
    pub fun: Vec<f64>,
    /// Number of iterations used
    pub iterations: usize,
    /// Norm of the residual
    pub residual_norm: f64,
    /// Whether the method converged
    pub converged: bool,
}

/// Newton's method for systems of nonlinear equations.
///
/// # Arguments
/// * `f` - Function F: R^n -> R^n to find root of
/// * `x0` - Initial guess
/// * `options` - Solver options
///
/// # Returns
/// Root of `F` (where F(x) ≈ 0)
///
/// # Note
/// Uses finite differences to approximate the Jacobian.
/// Has quadratic convergence near the root but may diverge if x0 is far from root.
pub fn newton_system<F>(f: F, x0: &[f64], options: &RootOptions) -> OptimizeResult<MultiRootResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "newton_system: empty initial guess".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);

    if fx.len() != n {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "newton_system: function returns {} values but input has {} dimensions",
                fx.len(),
                n
            ),
        });
    }

    for iter in 0..options.max_iter {
        let res_norm = norm(&fx);

        // Check convergence
        if res_norm < options.tol {
            return Ok(MultiRootResult {
                x,
                fun: fx,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: true,
            });
        }

        // Compute Jacobian
        let jacobian = finite_difference_jacobian(&f, &x, &fx, options.eps);

        // Solve J * dx = -F(x)
        let neg_fx: Vec<f64> = fx.iter().map(|v| -v).collect();
        let dx = match solve_linear_system(&jacobian, &neg_fx) {
            Some(dx) => dx,
            None => {
                return Err(OptimizeError::NumericalError {
                    message: "Singular Jacobian in newton_system".to_string(),
                });
            }
        };

        // Update x
        for i in 0..n {
            x[i] += dx[i];
        }

        // Check step size convergence
        if norm(&dx) < options.x_tol {
            fx = f(&x);
            return Ok(MultiRootResult {
                x,
                fun: fx.clone(),
                iterations: iter + 1,
                residual_norm: norm(&fx),
                converged: true,
            });
        }

        fx = f(&x);
    }

    Ok(MultiRootResult {
        x,
        fun: fx.clone(),
        iterations: options.max_iter,
        residual_norm: norm(&fx),
        converged: false,
    })
}

/// Broyden's method (rank-1 update) for systems of nonlinear equations.
///
/// # Arguments
/// * `f` - Function F: R^n -> R^n to find root of
/// * `x0` - Initial guess
/// * `options` - Solver options
///
/// # Returns
/// Root of `F` (where F(x) ≈ 0)
///
/// # Note
/// Broyden's method is a quasi-Newton method that approximates the Jacobian
/// using rank-1 updates. It requires fewer function evaluations than Newton's
/// method but may converge more slowly.
pub fn broyden1<F>(f: F, x0: &[f64], options: &RootOptions) -> OptimizeResult<MultiRootResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "broyden1: empty initial guess".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);

    if fx.len() != n {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "broyden1: function returns {} values but input has {} dimensions",
                fx.len(),
                n
            ),
        });
    }

    // Initialize Jacobian approximation with finite differences
    let mut jacobian = finite_difference_jacobian(&f, &x, &fx, options.eps);

    for iter in 0..options.max_iter {
        let res_norm = norm(&fx);

        // Check convergence
        if res_norm < options.tol {
            return Ok(MultiRootResult {
                x,
                fun: fx,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: true,
            });
        }

        // Solve J * dx = -F(x)
        let neg_fx: Vec<f64> = fx.iter().map(|v| -v).collect();
        let dx = match solve_linear_system(&jacobian, &neg_fx) {
            Some(dx) => dx,
            None => {
                // Re-initialize Jacobian if singular
                jacobian = finite_difference_jacobian(&f, &x, &fx, options.eps);
                match solve_linear_system(&jacobian, &neg_fx) {
                    Some(dx) => dx,
                    None => {
                        return Err(OptimizeError::NumericalError {
                            message: "Singular Jacobian in broyden1".to_string(),
                        });
                    }
                }
            }
        };

        // Update x
        let _x_old = x.clone();
        for i in 0..n {
            x[i] += dx[i];
        }

        // Check step size convergence
        if norm(&dx) < options.x_tol {
            fx = f(&x);
            return Ok(MultiRootResult {
                x,
                fun: fx.clone(),
                iterations: iter + 1,
                residual_norm: norm(&fx),
                converged: true,
            });
        }

        // Compute new function value
        let fx_new = f(&x);

        // Broyden rank-1 update: J_new = J + (df - J*dx) * dx^T / (dx^T * dx)
        // where df = F(x_new) - F(x_old)
        let df: Vec<f64> = fx_new.iter().zip(fx.iter()).map(|(a, b)| a - b).collect();

        // Compute J * dx
        let j_dx: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| jacobian[i][j] * dx[j]).sum())
            .collect();

        // Compute (df - J*dx)
        let diff: Vec<f64> = df.iter().zip(j_dx.iter()).map(|(a, b)| a - b).collect();

        // Compute dx^T * dx
        let dx_dot_dx: f64 = dx.iter().map(|v| v * v).sum();

        if dx_dot_dx > SINGULAR_THRESHOLD {
            // Update Jacobian: J[i][j] += diff[i] * dx[j] / dx_dot_dx
            for i in 0..n {
                for j in 0..n {
                    jacobian[i][j] += diff[i] * dx[j] / dx_dot_dx;
                }
            }
        }

        fx = fx_new;
    }

    Ok(MultiRootResult {
        x,
        fun: fx.clone(),
        iterations: options.max_iter,
        residual_norm: norm(&fx),
        converged: false,
    })
}

/// Levenberg-Marquardt algorithm for systems of nonlinear equations.
///
/// # Arguments
/// * `f` - Function F: R^n -> R^n to find root of
/// * `x0` - Initial guess
/// * `options` - Solver options
///
/// # Returns
/// Root of `F` (where F(x) ≈ 0)
///
/// # Note
/// Levenberg-Marquardt is a damped Newton method that interpolates between
/// Newton's method and gradient descent. It's more robust than Newton's method
/// for problems where the initial guess is far from the solution.
pub fn levenberg_marquardt<F>(
    f: F,
    x0: &[f64],
    options: &RootOptions,
) -> OptimizeResult<MultiRootResult>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "levenberg_marquardt: empty initial guess".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);

    if fx.len() != n {
        return Err(OptimizeError::InvalidInput {
            context: format!(
                "levenberg_marquardt: function returns {} values but input has {} dimensions",
                fx.len(),
                n
            ),
        });
    }

    let mut lambda = 0.001; // Damping parameter
    let lambda_up = 10.0;
    let lambda_down = 0.1;

    for iter in 0..options.max_iter {
        let res_norm = norm(&fx);

        // Check convergence
        if res_norm < options.tol {
            return Ok(MultiRootResult {
                x,
                fun: fx,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: true,
            });
        }

        // Compute Jacobian
        let jacobian = finite_difference_jacobian(&f, &x, &fx, options.eps);

        // Compute J^T * J + lambda * I
        let mut jtj: Vec<Vec<f64>> = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    jtj[i][j] += jacobian[k][i] * jacobian[k][j];
                }
                if i == j {
                    jtj[i][j] += lambda;
                }
            }
        }

        // Compute J^T * F(x)
        let jtf: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|k| jacobian[k][i] * fx[k]).sum::<f64>())
            .collect();

        // Solve (J^T*J + lambda*I) * dx = -J^T * F(x)
        let neg_jtf: Vec<f64> = jtf.iter().map(|v| -v).collect();
        let dx = match solve_linear_system(&jtj, &neg_jtf) {
            Some(dx) => dx,
            None => {
                lambda *= lambda_up;
                continue;
            }
        };

        // Try the step
        let x_new: Vec<f64> = x.iter().zip(dx.iter()).map(|(a, b)| a + b).collect();
        let fx_new = f(&x_new);
        let new_res_norm = norm(&fx_new);

        // Accept or reject step
        if new_res_norm < res_norm {
            x = x_new;
            fx = fx_new;
            lambda *= lambda_down;

            // Check step size convergence
            if norm(&dx) < options.x_tol {
                return Ok(MultiRootResult {
                    x,
                    fun: fx.clone(),
                    iterations: iter + 1,
                    residual_norm: norm(&fx),
                    converged: true,
                });
            }
        } else {
            lambda *= lambda_up;
        }

        // Prevent lambda from becoming too large or too small
        lambda = lambda.clamp(ZERO_THRESHOLD, 1e10);
    }

    Ok(MultiRootResult {
        x,
        fun: fx.clone(),
        iterations: options.max_iter,
        residual_norm: norm(&fx),
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_newton_system_linear() {
        // Solve: x + y = 3, 2x - y = 0
        // Solution: x = 1, y = 2
        let f = |x: &[f64]| vec![x[0] + x[1] - 3.0, 2.0 * x[0] - x[1]];

        let result =
            newton_system(f, &[0.0, 0.0], &RootOptions::default()).expect("newton_system failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-6);
        assert!((result.x[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_newton_system_quadratic() {
        // Solve: x^2 + y^2 = 1, x - y = 0
        // Solution: x = y = 1/sqrt(2) ≈ 0.707
        let f = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 1.0, x[0] - x[1]];

        let result =
            newton_system(f, &[0.5, 0.5], &RootOptions::default()).expect("newton_system failed");

        assert!(result.converged);
        let expected = 1.0 / (2.0_f64).sqrt();
        assert!((result.x[0] - expected).abs() < 1e-6);
        assert!((result.x[1] - expected).abs() < 1e-6);
    }

    #[test]
    fn test_newton_system_3d() {
        // Solve: x + y + z = 6, x - y = 0, y - z = 0
        // Solution: x = y = z = 2
        let f = |x: &[f64]| vec![x[0] + x[1] + x[2] - 6.0, x[0] - x[1], x[1] - x[2]];

        let result = newton_system(f, &[1.0, 1.0, 1.0], &RootOptions::default())
            .expect("newton_system failed");

        assert!(result.converged);
        assert!((result.x[0] - 2.0).abs() < 1e-6);
        assert!((result.x[1] - 2.0).abs() < 1e-6);
        assert!((result.x[2] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_broyden1_linear() {
        // Same linear system as newton test
        let f = |x: &[f64]| vec![x[0] + x[1] - 3.0, 2.0 * x[0] - x[1]];

        let result = broyden1(f, &[0.0, 0.0], &RootOptions::default()).expect("broyden1 failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-6);
        assert!((result.x[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_broyden1_nonlinear() {
        // Solve: x^2 - y = 0, x + y - 2 = 0
        // Solution: x = 1, y = 1
        let f = |x: &[f64]| vec![x[0] * x[0] - x[1], x[0] + x[1] - 2.0];

        let result = broyden1(f, &[0.5, 0.5], &RootOptions::default()).expect("broyden1 failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-5);
        assert!((result.x[1] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_levenberg_marquardt_linear() {
        // Same linear system
        let f = |x: &[f64]| vec![x[0] + x[1] - 3.0, 2.0 * x[0] - x[1]];

        let result = levenberg_marquardt(f, &[0.0, 0.0], &RootOptions::default())
            .expect("levenberg_marquardt failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-5);
        assert!((result.x[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_levenberg_marquardt_nonlinear() {
        // Nonlinear system that demonstrates LM's robustness
        // f1 = x^3 - y = 0
        // f2 = x + y^3 - 2 = 0
        // Has solution near (1, 1)
        let f = |x: &[f64]| vec![x[0] * x[0] * x[0] - x[1], x[0] + x[1] * x[1] * x[1] - 2.0];

        let result = levenberg_marquardt(f, &[0.5, 0.5], &RootOptions::default())
            .expect("levenberg_marquardt failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_newton_system_empty_input() {
        let f = |_: &[f64]| vec![];
        let result = newton_system(f, &[], &RootOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
    }

    #[test]
    fn test_broyden1_dimension_mismatch() {
        // Function returns different dimension than input
        let f = |x: &[f64]| vec![x[0], x[1], 0.0]; // Returns 3 values for 2D input

        let result = broyden1(f, &[1.0, 1.0], &RootOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
    }

    #[test]
    fn test_compare_methods() {
        // All methods should find similar solutions
        let f = |x: &[f64]| vec![x[0] * x[0] + x[1] * x[1] - 4.0, x[0] - x[1]];

        let newton_result =
            newton_system(&f, &[1.0, 1.0], &RootOptions::default()).expect("newton failed");
        let broyden_result =
            broyden1(&f, &[1.0, 1.0], &RootOptions::default()).expect("broyden failed");
        let lm_result =
            levenberg_marquardt(&f, &[1.0, 1.0], &RootOptions::default()).expect("lm failed");

        // All should converge to x = y = sqrt(2)
        let expected = (2.0_f64).sqrt();

        assert!(newton_result.converged);
        assert!((newton_result.x[0] - expected).abs() < 1e-5);

        assert!(broyden_result.converged);
        assert!((broyden_result.x[0] - expected).abs() < 1e-5);

        assert!(lm_result.converged);
        assert!((lm_result.x[0] - expected).abs() < 1e-4);
    }
}
