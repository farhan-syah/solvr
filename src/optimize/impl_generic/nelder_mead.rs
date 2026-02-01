//! Nelder-Mead simplex method for multivariate minimization.
//!
//! This implementation uses tensor operations throughout to leverage
//! SIMD on CPU and GPU acceleration when available.

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::minimize::MinimizeOptions;

use super::helpers::{TensorMinimizeResult, compare_f64_nan_safe};
use super::utils::SINGULAR_THRESHOLD;

/// Nelder-Mead simplex method for minimization using tensor operations.
///
/// The simplex is stored as a [n+1, n] tensor where each row is a vertex.
/// All vertex operations use tensor ops to stay on device.
pub fn nelder_mead_impl<R, C, F>(
    client: &C,
    f: F,
    x0: &Tensor<R>,
    options: &MinimizeOptions,
) -> OptimizeResult<TensorMinimizeResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
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

    // Initialize simplex as [n+1, n] tensor
    // First vertex is x0, others are x0 + delta*e_i
    let simplex = initialize_simplex(client, x0, n)?;

    // Evaluate function at all vertices - store as Vec<f64> for sorting
    // (sorting indices is inherently a CPU operation)
    let mut f_values = Vec::with_capacity(n + 1);
    let mut nfev = 0;
    for i in 0..=n {
        let vertex = extract_row(client, &simplex, i, n)?;
        let fval = f(&vertex).map_err(|e| OptimizeError::NumericalError {
            message: format!("nelder_mead: initial evaluation - {}", e),
        })?;
        f_values.push(fval);
        nfev += 1;
    }

    // Working storage for simplex vertices (updated in-place)
    let mut vertices: Vec<Tensor<R>> = Vec::with_capacity(n + 1);
    for i in 0..=n {
        vertices.push(extract_row(client, &simplex, i, n)?);
    }

    for iter in 0..options.max_iter {
        // Sort simplex by function values (NaN-safe)
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| compare_f64_nan_safe(f_values[a], f_values[b]));

        let best_idx = indices[0];
        let worst_idx = indices[n];
        let second_worst_idx = indices[n - 1];

        // Check for NaN in best value
        if f_values[best_idx].is_nan() {
            return Err(OptimizeError::NumericalError {
                message: "nelder_mead: all function values are NaN".to_string(),
            });
        }

        // Check convergence
        let f_range = f_values[worst_idx] - f_values[best_idx];
        if f_range < options.f_tol {
            return Ok(TensorMinimizeResult {
                x: vertices[best_idx].clone(),
                fun: f_values[best_idx],
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // Compute centroid of all vertices except worst
        let centroid = compute_centroid(client, &vertices, &indices[..n])?;

        // Reflection: reflected = centroid + alpha * (centroid - worst)
        let worst = &vertices[worst_idx];
        let reflected = reflect_point(client, &centroid, worst, alpha)?;
        let f_reflected = f(&reflected).map_err(|e| OptimizeError::NumericalError {
            message: format!("nelder_mead: reflection - {}", e),
        })?;
        nfev += 1;

        if f_reflected < f_values[second_worst_idx] && f_reflected >= f_values[best_idx] {
            // Accept reflection
            vertices[worst_idx] = reflected;
            f_values[worst_idx] = f_reflected;
            continue;
        }

        // Expansion
        if f_reflected < f_values[best_idx] {
            // expanded = centroid + gamma * (reflected - centroid)
            let expanded = reflect_point(client, &centroid, &reflected, -gamma)?;
            let f_expanded = f(&expanded).map_err(|e| OptimizeError::NumericalError {
                message: format!("nelder_mead: expansion - {}", e),
            })?;
            nfev += 1;

            if f_expanded < f_reflected {
                vertices[worst_idx] = expanded;
                f_values[worst_idx] = f_expanded;
            } else {
                vertices[worst_idx] = reflected;
                f_values[worst_idx] = f_reflected;
            }
            continue;
        }

        // Contraction
        let (contracted, f_contracted) = if f_reflected < f_values[worst_idx] {
            // Outside contraction: contracted = centroid + rho * (reflected - centroid)
            let contracted = reflect_point(client, &centroid, &reflected, -rho)?;
            let f_contracted = f(&contracted).map_err(|e| OptimizeError::NumericalError {
                message: format!("nelder_mead: outside contraction - {}", e),
            })?;
            nfev += 1;
            (contracted, f_contracted)
        } else {
            // Inside contraction: contracted = centroid + rho * (worst - centroid)
            let contracted = reflect_point(client, &centroid, worst, -rho)?;
            let f_contracted = f(&contracted).map_err(|e| OptimizeError::NumericalError {
                message: format!("nelder_mead: inside contraction - {}", e),
            })?;
            nfev += 1;
            (contracted, f_contracted)
        };

        if f_contracted < f_values[worst_idx].min(f_reflected) {
            vertices[worst_idx] = contracted;
            f_values[worst_idx] = f_contracted;
            continue;
        }

        // Shrink: move all vertices (except best) towards best
        let best = &vertices[best_idx].clone();
        for &idx in &indices[1..=n] {
            // new_vertex = best + sigma * (vertex - best)
            let diff = client
                .sub(&vertices[idx], best)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("nelder_mead: shrink diff - {}", e),
                })?;
            let scaled = client
                .mul_scalar(&diff, sigma)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("nelder_mead: shrink scale - {}", e),
                })?;
            vertices[idx] = client
                .add(best, &scaled)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("nelder_mead: shrink add - {}", e),
                })?;
            f_values[idx] = f(&vertices[idx]).map_err(|e| OptimizeError::NumericalError {
                message: format!("nelder_mead: shrink eval - {}", e),
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

    Ok(TensorMinimizeResult {
        x: vertices[best_idx].clone(),
        fun: f_values[best_idx],
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

/// Initialize simplex with n+1 vertices.
/// Returns a [n+1, n] tensor where row 0 is x0 and rows 1..n+1 are perturbations.
fn initialize_simplex<R, C>(
    client: &C,
    x0: &Tensor<R>,
    n: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    // Build simplex data: first row is x0, others are x0 + delta*e_i
    let x0_data: Vec<f64> = x0.to_vec();
    let mut simplex_data = Vec::with_capacity((n + 1) * n);

    // First vertex is x0
    simplex_data.extend_from_slice(&x0_data);

    // Other vertices are perturbations
    for i in 0..n {
        for (j, &x0j) in x0_data.iter().enumerate() {
            if i == j {
                let delta = if x0j.abs() > SINGULAR_THRESHOLD {
                    0.05 * x0j
                } else {
                    0.00025
                };
                simplex_data.push(x0j + delta);
            } else {
                simplex_data.push(x0j);
            }
        }
    }

    Ok(Tensor::<R>::from_slice(
        &simplex_data,
        &[n + 1, n],
        client.device(),
    ))
}

/// Extract row i from a [m, n] tensor as a [n] tensor.
fn extract_row<R, C>(
    client: &C,
    matrix: &Tensor<R>,
    row: usize,
    n: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let indices = Tensor::<R>::from_slice(&[row as i64], &[1], client.device());
    let row_2d = client
        .index_select(matrix, 0, &indices)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("nelder_mead: extract row - {}", e),
        })?;
    row_2d
        .reshape(&[n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("nelder_mead: reshape row - {}", e),
        })
}

/// Compute centroid of vertices at given indices.
fn compute_centroid<R, C>(
    client: &C,
    vertices: &[Tensor<R>],
    indices: &[usize],
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let k = indices.len();
    if k == 0 {
        return Err(OptimizeError::NumericalError {
            message: "nelder_mead: empty indices for centroid".to_string(),
        });
    }

    // Sum all vertices at given indices
    let mut sum = vertices[indices[0]].clone();
    for &idx in &indices[1..] {
        sum = client
            .add(&sum, &vertices[idx])
            .map_err(|e| OptimizeError::NumericalError {
                message: format!("nelder_mead: centroid sum - {}", e),
            })?;
    }

    // Divide by count
    client
        .mul_scalar(&sum, 1.0 / k as f64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("nelder_mead: centroid div - {}", e),
        })
}

/// Compute reflected point: result = base + coeff * (base - point)
/// For reflection: coeff = alpha (positive)
/// For expansion: coeff = -gamma (negative to go past reflected)
/// For contraction: coeff = -rho (negative to stay between)
fn reflect_point<R, C>(
    client: &C,
    base: &Tensor<R>,
    point: &Tensor<R>,
    coeff: f64,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    // diff = base - point
    let diff = client
        .sub(base, point)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("nelder_mead: reflect diff - {}", e),
        })?;
    // scaled = coeff * diff
    let scaled = client
        .mul_scalar(&diff, coeff)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("nelder_mead: reflect scale - {}", e),
        })?;
    // result = base + scaled
    client
        .add(base, &scaled)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("nelder_mead: reflect add - {}", e),
        })
}
