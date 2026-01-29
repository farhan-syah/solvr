//! Multivariate unconstrained minimization algorithms.
//!
//! This module provides methods for finding minima of scalar-valued functions
//! f: R^n -> R without constraints.

// Indexed loops are clearer for matrix/simplex operations
#![allow(clippy::needless_range_loop)]

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::utils::{
    SINGULAR_THRESHOLD, ZERO_THRESHOLD, finite_difference_gradient_forward, norm,
};

/// Options for multivariate minimization.
#[derive(Debug, Clone)]
pub struct MinimizeOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence (function value change)
    pub f_tol: f64,
    /// Tolerance for convergence (argument change)
    pub x_tol: f64,
    /// Tolerance for gradient norm (gradient-based methods)
    pub g_tol: f64,
    /// Step size for finite difference gradient approximation
    pub eps: f64,
}

impl Default for MinimizeOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            f_tol: 1e-8,
            x_tol: 1e-8,
            g_tol: 1e-8,
            eps: 1e-8,
        }
    }
}

/// Result from a multivariate minimization method.
#[derive(Debug, Clone)]
pub struct MultiMinimizeResult {
    /// The minimum point found
    pub x: Vec<f64>,
    /// Function value at minimum
    pub fun: f64,
    /// Number of iterations used
    pub iterations: usize,
    /// Number of function evaluations
    pub nfev: usize,
    /// Whether the method converged
    pub converged: bool,
}

/// Nelder-Mead simplex algorithm for derivative-free minimization.
///
/// # Arguments
/// * `f` - Function f: R^n -> R to minimize
/// * `x0` - Initial guess
/// * `options` - Solver options
///
/// # Returns
/// Minimum of `f`
///
/// # Note
/// Nelder-Mead is robust and doesn't require derivatives, but convergence
/// can be slow for high-dimensional problems (n > 10).
pub fn nelder_mead<F>(
    f: F,
    x0: &[f64],
    options: &MinimizeOptions,
) -> OptimizeResult<MultiMinimizeResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "nelder_mead: empty initial guess".to_string(),
        });
    }

    // Simplex parameters
    let alpha = 1.0; // Reflection
    let gamma = 2.0; // Expansion
    let rho = 0.5; // Contraction
    let sigma = 0.5; // Shrink

    // Initialize simplex with n+1 vertices
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());

    // Create remaining vertices by perturbing each dimension
    for i in 0..n {
        let mut vertex = x0.to_vec();
        let delta = if x0[i].abs() < ZERO_THRESHOLD {
            0.00025
        } else {
            0.05 * x0[i].abs()
        };
        vertex[i] += delta;
        simplex.push(vertex);
    }

    // Compute function values at all vertices
    let mut f_values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
    let mut nfev = n + 1;

    for iter in 0..options.max_iter {
        // Sort vertices by function value
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| f_values[a].partial_cmp(&f_values[b]).unwrap());

        // Reorder simplex and f_values
        let sorted_simplex: Vec<Vec<f64>> = indices.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_f: Vec<f64> = indices.iter().map(|&i| f_values[i]).collect();
        simplex = sorted_simplex;
        f_values = sorted_f;

        // Check convergence
        let f_best = f_values[0];
        let f_worst = f_values[n];
        let f_range = (f_worst - f_best).abs();

        // Compute simplex diameter
        let mut max_dist = 0.0_f64;
        for i in 1..=n {
            let dist: f64 = simplex[0]
                .iter()
                .zip(simplex[i].iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            max_dist = max_dist.max(dist);
        }

        if f_range < options.f_tol && max_dist < options.x_tol {
            return Ok(MultiMinimizeResult {
                x: simplex[0].clone(),
                fun: f_values[0],
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute centroid of all vertices except worst
        let mut centroid = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                centroid[j] += simplex[i][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        // Reflection
        let mut x_r = vec![0.0; n];
        for j in 0..n {
            x_r[j] = centroid[j] + alpha * (centroid[j] - simplex[n][j]);
        }
        let f_r = f(&x_r);
        nfev += 1;

        if f_r < f_values[0] {
            // Try expansion
            let mut x_e = vec![0.0; n];
            for j in 0..n {
                x_e[j] = centroid[j] + gamma * (x_r[j] - centroid[j]);
            }
            let f_e = f(&x_e);
            nfev += 1;

            if f_e < f_r {
                simplex[n] = x_e;
                f_values[n] = f_e;
            } else {
                simplex[n] = x_r;
                f_values[n] = f_r;
            }
        } else if f_r < f_values[n - 1] {
            // Accept reflection
            simplex[n] = x_r;
            f_values[n] = f_r;
        } else {
            // Contraction
            let (x_c, f_c) = if f_r < f_values[n] {
                // Outside contraction
                let mut x_c = vec![0.0; n];
                for j in 0..n {
                    x_c[j] = centroid[j] + rho * (x_r[j] - centroid[j]);
                }
                let f_c = f(&x_c);
                nfev += 1;
                (x_c, f_c)
            } else {
                // Inside contraction
                let mut x_c = vec![0.0; n];
                for j in 0..n {
                    x_c[j] = centroid[j] - rho * (centroid[j] - simplex[n][j]);
                }
                let f_c = f(&x_c);
                nfev += 1;
                (x_c, f_c)
            };

            if f_c < f_values[n].min(f_r) {
                simplex[n] = x_c;
                f_values[n] = f_c;
            } else {
                // Shrink
                for i in 1..=n {
                    for j in 0..n {
                        simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                    }
                    f_values[i] = f(&simplex[i]);
                    nfev += 1;
                }
            }
        }
    }

    // Return best result even if not converged
    let mut indices: Vec<usize> = (0..=n).collect();
    indices.sort_by(|&a, &b| f_values[a].partial_cmp(&f_values[b]).unwrap());

    Ok(MultiMinimizeResult {
        x: simplex[indices[0]].clone(),
        fun: f_values[indices[0]],
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

/// Powell's direction set method for derivative-free minimization.
///
/// # Arguments
/// * `f` - Function f: R^n -> R to minimize
/// * `x0` - Initial guess
/// * `options` - Solver options
///
/// # Returns
/// Minimum of `f`
///
/// # Note
/// Powell's method performs successive line searches along conjugate directions.
/// It doesn't require derivatives but can be faster than Nelder-Mead.
pub fn powell<F>(f: F, x0: &[f64], options: &MinimizeOptions) -> OptimizeResult<MultiMinimizeResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "powell: empty initial guess".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut nfev = 1;

    // Initialize direction set to coordinate directions
    let mut directions: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut d = vec![0.0; n];
            d[i] = 1.0;
            d
        })
        .collect();

    for iter in 0..options.max_iter {
        let x_start = x.clone();
        let fx_start = fx;

        // Perform line search along each direction
        let mut max_decrease = 0.0;
        let mut max_decrease_idx = 0;

        for (i, dir) in directions.iter().enumerate() {
            let (x_new, fx_new, evals) = line_search_quadratic(&f, &x, dir, fx);
            nfev += evals;

            let decrease = fx - fx_new;
            if decrease > max_decrease {
                max_decrease = decrease;
                max_decrease_idx = i;
            }

            x = x_new;
            fx = fx_new;
        }

        // Check convergence
        let dx: f64 = x
            .iter()
            .zip(x_start.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();

        if dx < options.x_tol || (fx_start - fx).abs() < options.f_tol {
            return Ok(MultiMinimizeResult {
                x,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute new direction
        let mut new_dir: Vec<f64> = x.iter().zip(x_start.iter()).map(|(a, b)| a - b).collect();
        let dir_norm = norm(&new_dir);
        if dir_norm > SINGULAR_THRESHOLD {
            for d in &mut new_dir {
                *d /= dir_norm;
            }

            // Line search along new direction
            let (x_new, fx_new, evals) = line_search_quadratic(&f, &x, &new_dir, fx);
            nfev += evals;

            // Replace direction with largest decrease
            if fx_new < fx {
                directions[max_decrease_idx] = new_dir;
                x = x_new;
                fx = fx_new;
            }
        }
    }

    Ok(MultiMinimizeResult {
        x,
        fun: fx,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

/// Simple quadratic line search.
/// Returns (new_x, new_fx, num_evaluations).
fn line_search_quadratic<F>(f: &F, x: &[f64], direction: &[f64], fx: f64) -> (Vec<f64>, f64, usize)
where
    F: Fn(&[f64]) -> f64,
{
    let _n = x.len();
    let mut nfev = 0;

    // Bracket the minimum
    let alpha = 0.0;
    let f_alpha = fx;

    let mut beta = 1.0;
    let x_beta: Vec<f64> = x
        .iter()
        .zip(direction.iter())
        .map(|(a, d)| a + beta * d)
        .collect();
    let mut f_beta = f(&x_beta);
    nfev += 1;

    // If f_beta > f_alpha, reduce step size
    while f_beta > f_alpha && beta > ZERO_THRESHOLD {
        beta *= 0.5;
        let x_beta: Vec<f64> = x
            .iter()
            .zip(direction.iter())
            .map(|(a, d)| a + beta * d)
            .collect();
        f_beta = f(&x_beta);
        nfev += 1;
    }

    // If still not better, try negative direction
    if f_beta >= f_alpha {
        beta = -1.0;
        let x_beta: Vec<f64> = x
            .iter()
            .zip(direction.iter())
            .map(|(a, d)| a + beta * d)
            .collect();
        f_beta = f(&x_beta);
        nfev += 1;

        while f_beta > f_alpha && beta.abs() > ZERO_THRESHOLD {
            beta *= 0.5;
            let x_beta: Vec<f64> = x
                .iter()
                .zip(direction.iter())
                .map(|(a, d)| a + beta * d)
                .collect();
            f_beta = f(&x_beta);
            nfev += 1;
        }
    }

    if f_beta >= f_alpha {
        return (x.to_vec(), fx, nfev);
    }

    // Golden section search to refine
    let inv_phi = ((5.0_f64).sqrt() - 1.0) / 2.0;
    let inv_phi2 = 1.0 - inv_phi;

    let (mut a, mut b) = if beta > 0.0 {
        (alpha, beta)
    } else {
        (beta, alpha)
    };

    let mut x1 = a + inv_phi2 * (b - a);
    let mut x2 = a + inv_phi * (b - a);

    let x_at = |t: f64| -> Vec<f64> {
        x.iter()
            .zip(direction.iter())
            .map(|(a, d)| a + t * d)
            .collect()
    };

    let mut f1 = f(&x_at(x1));
    let mut f2 = f(&x_at(x2));
    nfev += 2;

    for _ in 0..20 {
        if (b - a).abs() < 1e-8 {
            break;
        }

        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + inv_phi2 * (b - a);
            f1 = f(&x_at(x1));
            nfev += 1;
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + inv_phi * (b - a);
            f2 = f(&x_at(x2));
            nfev += 1;
        }
    }

    let best_alpha = 0.5 * (a + b);
    let x_best = x_at(best_alpha);
    let f_best = f(&x_best);
    nfev += 1;

    (x_best, f_best, nfev)
}

/// BFGS quasi-Newton method for minimization.
///
/// # Arguments
/// * `f` - Function f: R^n -> R to minimize
/// * `x0` - Initial guess
/// * `options` - Solver options
///
/// # Returns
/// Minimum of `f`
///
/// # Note
/// BFGS uses gradient information and maintains an approximation of the
/// inverse Hessian. It has superlinear convergence near the minimum.
/// Uses finite differences for gradient approximation.
pub fn bfgs<F>(f: F, x0: &[f64], options: &MinimizeOptions) -> OptimizeResult<MultiMinimizeResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "bfgs: empty initial guess".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut nfev = 1;

    let mut grad = finite_difference_gradient_forward(&f, &x, fx, options.eps);
    nfev += n;

    // Initialize inverse Hessian approximation to identity
    let mut h_inv: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; n];
            row[i] = 1.0;
            row
        })
        .collect();

    for iter in 0..options.max_iter {
        // Check gradient convergence
        let grad_norm = norm(&grad);
        if grad_norm < options.g_tol {
            return Ok(MultiMinimizeResult {
                x,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute search direction: p = -H_inv * grad
        let mut p = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                p[i] -= h_inv[i][j] * grad[j];
            }
        }

        // Line search with Armijo condition
        let (_alpha, x_new, fx_new, evals) = backtracking_line_search(&f, &x, &p, fx, &grad);
        nfev += evals;

        // Check convergence
        let dx = norm(
            &x_new
                .iter()
                .zip(x.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<f64>>(),
        );
        if dx < options.x_tol || (fx - fx_new).abs() < options.f_tol {
            return Ok(MultiMinimizeResult {
                x: x_new,
                fun: fx_new,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute new gradient
        let grad_new = finite_difference_gradient_forward(&f, &x_new, fx_new, options.eps);
        nfev += n;

        // BFGS update
        // s = x_new - x
        let s: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        // y = grad_new - grad
        let y: Vec<f64> = grad_new
            .iter()
            .zip(grad.iter())
            .map(|(a, b)| a - b)
            .collect();

        // rho = 1 / (y^T * s)
        let ys: f64 = y.iter().zip(s.iter()).map(|(a, b)| a * b).sum();
        if ys.abs() > SINGULAR_THRESHOLD {
            let rho = 1.0 / ys;

            // H_new = (I - rho*s*y^T) * H * (I - rho*y*s^T) + rho*s*s^T
            // Simplified BFGS update
            let mut h_y = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    h_y[i] += h_inv[i][j] * y[j];
                }
            }

            let yhy: f64 = y.iter().zip(h_y.iter()).map(|(a, b)| a * b).sum();

            for i in 0..n {
                for j in 0..n {
                    h_inv[i][j] += rho * (1.0 + rho * yhy) * s[i] * s[j]
                        - rho * (s[i] * h_y[j] + h_y[i] * s[j]);
                }
            }
        }

        x = x_new;
        fx = fx_new;
        grad = grad_new;
    }

    Ok(MultiMinimizeResult {
        x,
        fun: fx,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

/// Backtracking line search with Armijo condition.
/// Returns (step_size, new_x, new_fx, num_evaluations).
fn backtracking_line_search<F>(
    f: &F,
    x: &[f64],
    p: &[f64],
    fx: f64,
    grad: &[f64],
) -> (f64, Vec<f64>, f64, usize)
where
    F: Fn(&[f64]) -> f64,
{
    let c = 0.0001; // Armijo constant
    let rho = 0.5; // Step reduction factor

    let grad_dot_p: f64 = grad.iter().zip(p.iter()).map(|(g, d)| g * d).sum();

    let mut alpha = 1.0;
    let mut nfev = 0;

    for _ in 0..50 {
        let x_new: Vec<f64> = x.iter().zip(p.iter()).map(|(a, d)| a + alpha * d).collect();
        let fx_new = f(&x_new);
        nfev += 1;

        // Armijo condition
        if fx_new <= fx + c * alpha * grad_dot_p {
            return (alpha, x_new, fx_new, nfev);
        }

        alpha *= rho;
    }

    // Return current point if line search fails
    (0.0, x.to_vec(), fx, nfev)
}

/// Conjugate gradient method for minimization.
///
/// # Arguments
/// * `f` - Function f: R^n -> R to minimize
/// * `x0` - Initial guess
/// * `options` - Solver options
///
/// # Returns
/// Minimum of `f`
///
/// # Note
/// Uses Polak-Ribière conjugate gradient with restarts.
/// Requires less memory than BFGS (no Hessian approximation).
pub fn conjugate_gradient<F>(
    f: F,
    x0: &[f64],
    options: &MinimizeOptions,
) -> OptimizeResult<MultiMinimizeResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "conjugate_gradient: empty initial guess".to_string(),
        });
    }

    let mut x = x0.to_vec();
    let mut fx = f(&x);
    let mut nfev = 1;

    let mut grad = finite_difference_gradient_forward(&f, &x, fx, options.eps);
    nfev += n;

    // Initial direction is negative gradient
    let mut p: Vec<f64> = grad.iter().map(|g| -g).collect();
    let mut grad_norm_sq: f64 = grad.iter().map(|g| g * g).sum();

    for iter in 0..options.max_iter {
        // Check gradient convergence
        if grad_norm_sq.sqrt() < options.g_tol {
            return Ok(MultiMinimizeResult {
                x,
                fun: fx,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Line search
        let (_alpha, x_new, fx_new, evals) = backtracking_line_search(&f, &x, &p, fx, &grad);
        nfev += evals;

        // Check convergence
        let dx = norm(
            &x_new
                .iter()
                .zip(x.iter())
                .map(|(a, b)| a - b)
                .collect::<Vec<f64>>(),
        );
        if dx < options.x_tol || (fx - fx_new).abs() < options.f_tol {
            return Ok(MultiMinimizeResult {
                x: x_new,
                fun: fx_new,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute new gradient
        let grad_new = finite_difference_gradient_forward(&f, &x_new, fx_new, options.eps);
        nfev += n;

        // Polak-Ribière beta
        let grad_new_norm_sq: f64 = grad_new.iter().map(|g| g * g).sum();
        let grad_diff_dot_new: f64 = grad_new
            .iter()
            .zip(grad.iter())
            .map(|(gn, g)| gn * (gn - g))
            .sum();

        let beta = (grad_diff_dot_new / grad_norm_sq).max(0.0);

        // Update direction: p = -grad_new + beta * p
        for i in 0..n {
            p[i] = -grad_new[i] + beta * p[i];
        }

        // Restart if direction is not descent
        let grad_dot_p: f64 = grad_new.iter().zip(p.iter()).map(|(g, d)| g * d).sum();
        if grad_dot_p >= 0.0 {
            for i in 0..n {
                p[i] = -grad_new[i];
            }
        }

        x = x_new;
        fx = fx_new;
        grad = grad_new;
        grad_norm_sq = grad_new_norm_sq;
    }

    Ok(MultiMinimizeResult {
        x,
        fun: fx,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rosenbrock(x: &[f64]) -> f64 {
        let a = 1.0;
        let b = 100.0;
        (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2)
    }

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    fn quadratic_2d(x: &[f64]) -> f64 {
        (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2)
    }

    #[test]
    fn test_nelder_mead_sphere() {
        let result = nelder_mead(sphere, &[1.0, 1.0, 1.0], &MinimizeOptions::default())
            .expect("nelder_mead failed");

        assert!(result.converged);
        assert!(result.fun < 1e-8);
        for xi in &result.x {
            assert!(xi.abs() < 1e-4);
        }
    }

    #[test]
    fn test_nelder_mead_quadratic() {
        let result = nelder_mead(quadratic_2d, &[0.0, 0.0], &MinimizeOptions::default())
            .expect("nelder_mead failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_nelder_mead_rosenbrock() {
        let mut opts = MinimizeOptions::default();
        opts.max_iter = 2000;

        let result = nelder_mead(rosenbrock, &[0.0, 0.0], &opts).expect("nelder_mead failed");

        // Rosenbrock is challenging, just check it gets close
        assert!((result.x[0] - 1.0).abs() < 0.1);
        assert!((result.x[1] - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_powell_sphere() {
        let result =
            powell(sphere, &[1.0, 1.0, 1.0], &MinimizeOptions::default()).expect("powell failed");

        assert!(result.converged);
        assert!(result.fun < 1e-6);
    }

    #[test]
    fn test_powell_quadratic() {
        let result =
            powell(quadratic_2d, &[0.0, 0.0], &MinimizeOptions::default()).expect("powell failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_bfgs_sphere() {
        let result =
            bfgs(sphere, &[1.0, 1.0, 1.0], &MinimizeOptions::default()).expect("bfgs failed");

        assert!(result.converged);
        assert!(result.fun < 1e-8);
    }

    #[test]
    fn test_bfgs_quadratic() {
        let result =
            bfgs(quadratic_2d, &[0.0, 0.0], &MinimizeOptions::default()).expect("bfgs failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-5);
        assert!((result.x[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_bfgs_rosenbrock() {
        let mut opts = MinimizeOptions::default();
        opts.max_iter = 500;

        let result = bfgs(rosenbrock, &[0.0, 0.0], &opts).expect("bfgs failed");

        assert!((result.x[0] - 1.0).abs() < 0.01);
        assert!((result.x[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_cg_sphere() {
        let result = conjugate_gradient(sphere, &[1.0, 1.0, 1.0], &MinimizeOptions::default())
            .expect("cg failed");

        assert!(result.converged);
        assert!(result.fun < 1e-6);
    }

    #[test]
    fn test_cg_quadratic() {
        let result = conjugate_gradient(quadratic_2d, &[0.0, 0.0], &MinimizeOptions::default())
            .expect("cg failed");

        assert!(result.converged);
        assert!((result.x[0] - 1.0).abs() < 1e-4);
        assert!((result.x[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_empty_input() {
        let result = nelder_mead(sphere, &[], &MinimizeOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));

        let result = powell(sphere, &[], &MinimizeOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));

        let result = bfgs(sphere, &[], &MinimizeOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
    }

    #[test]
    fn test_compare_methods() {
        // All methods should find similar solutions for a simple problem
        let f = |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2);
        let x0 = &[0.0, 0.0];

        let nm_result = nelder_mead(&f, x0, &MinimizeOptions::default()).expect("nm failed");
        let powell_result = powell(&f, x0, &MinimizeOptions::default()).expect("powell failed");
        let bfgs_result = bfgs(&f, x0, &MinimizeOptions::default()).expect("bfgs failed");
        let cg_result = conjugate_gradient(&f, x0, &MinimizeOptions::default()).expect("cg failed");

        // All should find minimum at (3, -1)
        assert!((nm_result.x[0] - 3.0).abs() < 1e-3);
        assert!((nm_result.x[1] - (-1.0)).abs() < 1e-3);

        assert!((powell_result.x[0] - 3.0).abs() < 1e-3);
        assert!((powell_result.x[1] - (-1.0)).abs() < 1e-3);

        assert!((bfgs_result.x[0] - 3.0).abs() < 1e-4);
        assert!((bfgs_result.x[1] - (-1.0)).abs() < 1e-4);

        assert!((cg_result.x[0] - 3.0).abs() < 1e-3);
        assert!((cg_result.x[1] - (-1.0)).abs() < 1e-3);
    }
}
