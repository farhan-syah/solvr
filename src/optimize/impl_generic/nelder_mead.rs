//! Nelder-Mead simplex method for multivariate minimization.

use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::MinimizeOptions;

use super::helpers::{TensorMinimizeResult, compare_f64_nan_safe};
use super::utils::SINGULAR_THRESHOLD;

/// Nelder-Mead simplex method for minimization using tensors.
pub fn nelder_mead_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    options: &MinimizeOptions,
) -> OptimizeResult<TensorMinimizeResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let n = x0.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "nelder_mead: empty initial guess".to_string(),
        });
    }

    // Nelder-Mead parameters
    let alpha = 1.0; // Reflection
    let gamma = 2.0; // Expansion
    let rho = 0.5; // Contraction
    let sigma = 0.5; // Shrink

    // Initialize simplex
    let x0_data: Vec<f64> = x0.to_vec();
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0_data.clone());

    for i in 0..n {
        let mut point = x0_data.clone();
        point[i] += if point[i].abs() > SINGULAR_THRESHOLD {
            0.05 * point[i]
        } else {
            0.00025
        };
        simplex.push(point);
    }

    // Evaluate function at simplex vertices
    let mut f_values: Vec<f64> = Vec::with_capacity(n + 1);
    let mut nfev = 0;
    for point in &simplex {
        let pt = Tensor::<R>::from_slice(point, &[n], client.device());
        let fval = f(&pt).map_err(|e| OptimizeError::NumericalError {
            message: format!("nelder_mead: initial evaluation - {}", e),
        })?;
        f_values.push(fval);
        nfev += 1;
    }

    for iter in 0..options.max_iter {
        // Sort simplex by function values (NaN-safe)
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| compare_f64_nan_safe(f_values[a], f_values[b]));

        let best_idx = indices[0];
        let worst_idx = indices[n];
        let second_worst_idx = indices[n - 1];

        // Check for NaN in best value - indicates numerical failure
        if f_values[best_idx].is_nan() {
            return Err(OptimizeError::NumericalError {
                message: "nelder_mead: all function values are NaN".to_string(),
            });
        }

        // Check convergence
        let f_range = f_values[worst_idx] - f_values[best_idx];
        if f_range < options.f_tol {
            let x = Tensor::<R>::from_slice(&simplex[best_idx], &[n], client.device());
            return Ok(TensorMinimizeResult {
                x,
                fun: f_values[best_idx],
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute centroid (excluding worst point)
        let mut centroid = vec![0.0; n];
        for &idx in &indices[..n] {
            for j in 0..n {
                centroid[j] += simplex[idx][j];
            }
        }
        for c in centroid.iter_mut() {
            *c /= n as f64;
        }

        // Reflection
        let mut reflected = vec![0.0; n];
        for j in 0..n {
            reflected[j] = centroid[j] + alpha * (centroid[j] - simplex[worst_idx][j]);
        }
        let reflected_tensor = Tensor::<R>::from_slice(&reflected, &[n], client.device());
        let f_reflected = f(&reflected_tensor).map_err(|e| OptimizeError::NumericalError {
            message: format!("nelder_mead: reflection - {}", e),
        })?;
        nfev += 1;

        if f_reflected < f_values[second_worst_idx] && f_reflected >= f_values[best_idx] {
            simplex[worst_idx] = reflected;
            f_values[worst_idx] = f_reflected;
            continue;
        }

        // Expansion
        if f_reflected < f_values[best_idx] {
            let mut expanded = vec![0.0; n];
            for j in 0..n {
                expanded[j] = centroid[j] + gamma * (reflected[j] - centroid[j]);
            }
            let expanded_tensor = Tensor::<R>::from_slice(&expanded, &[n], client.device());
            let f_expanded = f(&expanded_tensor).map_err(|e| OptimizeError::NumericalError {
                message: format!("nelder_mead: expansion - {}", e),
            })?;
            nfev += 1;

            if f_expanded < f_reflected {
                simplex[worst_idx] = expanded;
                f_values[worst_idx] = f_expanded;
            } else {
                simplex[worst_idx] = reflected;
                f_values[worst_idx] = f_reflected;
            }
            continue;
        }

        // Contraction
        let mut contracted = vec![0.0; n];
        if f_reflected < f_values[worst_idx] {
            // Outside contraction
            for j in 0..n {
                contracted[j] = centroid[j] + rho * (reflected[j] - centroid[j]);
            }
        } else {
            // Inside contraction
            for j in 0..n {
                contracted[j] = centroid[j] + rho * (simplex[worst_idx][j] - centroid[j]);
            }
        }
        let contracted_tensor = Tensor::<R>::from_slice(&contracted, &[n], client.device());
        let f_contracted = f(&contracted_tensor).map_err(|e| OptimizeError::NumericalError {
            message: format!("nelder_mead: contraction - {}", e),
        })?;
        nfev += 1;

        if f_contracted < f_values[worst_idx].min(f_reflected) {
            simplex[worst_idx] = contracted;
            f_values[worst_idx] = f_contracted;
            continue;
        }

        // Shrink
        for &idx in &indices[1..=n] {
            let best_point = simplex[best_idx].clone();
            for (j, val) in simplex[idx].iter_mut().enumerate() {
                *val = best_point[j] + sigma * (*val - best_point[j]);
            }
            let pt = Tensor::<R>::from_slice(&simplex[idx], &[n], client.device());
            f_values[idx] = f(&pt).map_err(|e| OptimizeError::NumericalError {
                message: format!("nelder_mead: shrink - {}", e),
            })?;
            nfev += 1;
        }
    }

    // Return best point found
    let mut best_idx = 0;
    for i in 1..=n {
        if f_values[i] < f_values[best_idx] {
            best_idx = i;
        }
    }
    let x = Tensor::<R>::from_slice(&simplex[best_idx], &[n], client.device());

    Ok(TensorMinimizeResult {
        x,
        fun: f_values[best_idx],
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}
