//! Tensor-based differential evolution implementation.
//!
//! Population stored as 2D tensor [pop_size, n]. All operations on device.

use numr::dtype::DType;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::global::GlobalOptions;

use super::TensorGlobalResult;

/// Differential Evolution global optimizer using tensor operations.
///
/// Population is stored as a 2D tensor [pop_size, n] on device.
pub fn differential_evolution_impl<R, C, F>(
    client: &C,
    f: F,
    lower_bounds: &Tensor<R>,
    upper_bounds: &Tensor<R>,
    options: &GlobalOptions,
) -> OptimizeResult<TensorGlobalResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let n = lower_bounds.shape()[0];
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "differential_evolution: empty bounds".to_string(),
        });
    }

    // Validate bounds once at start
    validate_bounds(lower_bounds, upper_bounds)?;

    // DE parameters
    let pop_size = (15 * n).max(25);
    let f_scale = 0.8;
    let cr = 0.9;

    // Compute bounds range (stays on device)
    let bounds_range = client
        .sub(upper_bounds, lower_bounds)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: bounds range - {}", e),
        })?;

    // Initialize population as 2D tensor [pop_size, n]
    // Each row is lower + rand * range
    let mut population = init_population(client, lower_bounds, &bounds_range, pop_size, n)?;

    // Evaluate initial population - must iterate (function returns scalar)
    let mut fitness = evaluate_population(client, &f, &population, pop_size, n)?;
    let mut nfev = pop_size;

    // Find best individual
    let (mut best_idx, mut best_fitness) = find_best(&fitness);

    for iter in 0..options.max_iter {
        // Check convergence
        let fitness_range = compute_fitness_range(&fitness);
        if fitness_range < options.tol {
            let x_best = extract_individual(client, &population, best_idx, n)?;
            return Ok(TensorGlobalResult {
                x: x_best,
                fun: best_fitness,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }

        // DE iteration: for each individual, create trial and possibly replace
        for (i, fit) in fitness.iter_mut().enumerate() {
            // Select three distinct random individuals (not i)
            let (r0, r1, r2) = select_random_indices(client, pop_size, i)?;

            // Extract individuals for mutation
            let x_r0 = extract_individual(client, &population, r0, n)?;
            let x_r1 = extract_individual(client, &population, r1, n)?;
            let x_r2 = extract_individual(client, &population, r2, n)?;
            let x_i = extract_individual(client, &population, i, n)?;

            // Mutant: x_r0 + f_scale * (x_r1 - x_r2)
            let diff = client
                .sub(&x_r1, &x_r2)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("de: diff - {}", e),
                })?;
            let scaled_diff = client
                .mul_scalar(&diff, f_scale)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("de: scaled diff - {}", e),
                })?;
            let mutant_unclamped = client
                .add(&x_r0, &scaled_diff)
                .map_err(|e| OptimizeError::NumericalError {
                    message: format!("de: mutant - {}", e),
                })?;

            // Clamp to bounds
            let mutant = clamp_to_bounds(client, &mutant_unclamped, lower_bounds, upper_bounds)?;

            // Crossover: create trial vector
            let trial = crossover(client, &x_i, &mutant, cr, n)?;

            // Evaluate trial
            let trial_fitness = f(&trial).map_err(|e| OptimizeError::NumericalError {
                message: format!("de: evaluation - {}", e),
            })?;
            nfev += 1;

            // Selection: if trial is better, replace
            if trial_fitness <= *fit {
                update_population(client, &mut population, i, &trial, n)?;
                *fit = trial_fitness;

                if trial_fitness < best_fitness {
                    best_fitness = trial_fitness;
                    best_idx = i;
                }
            }
        }
    }

    let x_best = extract_individual(client, &population, best_idx, n)?;
    Ok(TensorGlobalResult {
        x: x_best,
        fun: best_fitness,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

/// Initialize population tensor [pop_size, n] with uniform random within bounds.
fn init_population<R, C>(
    client: &C,
    lower: &Tensor<R>,
    range: &Tensor<R>,
    pop_size: usize,
    n: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    // Generate random [pop_size, n] in [0, 1)
    let rand_pop = client
        .rand(&[pop_size, n], DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: rand pop - {}", e),
        })?;

    // Broadcast lower and range to [pop_size, n]
    let lower_broadcast = lower
        .broadcast_to(&[pop_size, n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: broadcast lower - {}", e),
        })?;
    let range_broadcast = range
        .broadcast_to(&[pop_size, n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: broadcast range - {}", e),
        })?;

    // population = lower + rand * range
    let scaled = client
        .mul(&rand_pop, &range_broadcast)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: scale pop - {}", e),
        })?;
    client
        .add(&lower_broadcast, &scaled)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: init pop - {}", e),
        })
}

/// Evaluate all individuals in population.
fn evaluate_population<R, C, F>(
    client: &C,
    f: &F,
    population: &Tensor<R>,
    pop_size: usize,
    n: usize,
) -> OptimizeResult<Vec<f64>>
where
    R: Runtime,
    C: RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<f64>,
{
    let mut fitness = Vec::with_capacity(pop_size);
    for i in 0..pop_size {
        let individual = extract_individual(client, population, i, n)?;
        let fit = f(&individual).map_err(|e| OptimizeError::NumericalError {
            message: format!("de: initial evaluation - {}", e),
        })?;
        fitness.push(fit);
    }
    Ok(fitness)
}

/// Extract individual i from population tensor.
fn extract_individual<R, C>(
    _client: &C,
    population: &Tensor<R>,
    i: usize,
    n: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    // Narrow row i from [pop_size, n] -> [1, n], make contiguous, then reshape to [n]
    population
        .narrow(0, i, 1)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: narrow individual - {}", e),
        })?
        .contiguous()
        .reshape(&[n])
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: reshape individual - {}", e),
        })
}

/// Update population row i with new individual.
fn update_population<R, C>(
    _client: &C,
    population: &mut Tensor<R>,
    i: usize,
    individual: &Tensor<R>,
    n: usize,
) -> OptimizeResult<()>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    // Get current population data, update row, create new tensor
    // TODO: numr should have scatter/index_put for in-place updates
    let mut data: Vec<f64> = population.to_vec();
    let ind_data: Vec<f64> = individual.to_vec();

    for j in 0..n {
        data[i * n + j] = ind_data[j];
    }

    let pop_size = population.shape()[0];
    *population = Tensor::<R>::from_slice(&data, &[pop_size, n], population.device());
    Ok(())
}

/// Binomial crossover between target and mutant.
fn crossover<R, C>(
    client: &C,
    target: &Tensor<R>,
    mutant: &Tensor<R>,
    cr: f64,
    n: usize,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    // Generate random mask
    let rand_mask = client
        .rand(&[n], DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: crossover rand - {}", e),
        })?;

    // Ensure at least one from mutant (j_rand)
    let rand_idx = client
        .rand(&[1], DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: j_rand - {}", e),
        })?;
    let j_rand_data: Vec<f64> = rand_idx.to_vec();
    let j_rand = (j_rand_data[0] * n as f64) as usize;

    // Build trial: if rand < cr or j == j_rand, use mutant, else use target
    let mask_data: Vec<f64> = rand_mask.to_vec();
    let target_data: Vec<f64> = target.to_vec();
    let mutant_data: Vec<f64> = mutant.to_vec();

    let trial_data: Vec<f64> = (0..n)
        .map(|j| {
            if mask_data[j] < cr || j == j_rand {
                mutant_data[j]
            } else {
                target_data[j]
            }
        })
        .collect();

    Ok(Tensor::<R>::from_slice(
        &trial_data,
        &[n],
        client.device(),
    ))
}

/// Select 3 distinct random indices, none equal to i.
fn select_random_indices<R, C>(
    client: &C,
    pop_size: usize,
    exclude: usize,
) -> OptimizeResult<(usize, usize, usize)>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
{
    // Generate enough random values
    let rand_vals = client
        .rand(&[10], DType::F64)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: select random - {}", e),
        })?;
    let vals: Vec<f64> = rand_vals.to_vec();

    let mut selected = Vec::with_capacity(3);
    let mut idx = 0;

    while selected.len() < 3 && idx < 10 {
        let candidate = (vals[idx] * pop_size as f64) as usize % pop_size;
        if candidate != exclude && !selected.contains(&candidate) {
            selected.push(candidate);
        }
        idx += 1;
    }

    if selected.len() < 3 {
        // Fallback: deterministic selection
        for k in 0..pop_size {
            if k != exclude && !selected.contains(&k) {
                selected.push(k);
                if selected.len() >= 3 {
                    break;
                }
            }
        }
    }

    Ok((selected[0], selected[1], selected[2]))
}

fn find_best(fitness: &[f64]) -> (usize, f64) {
    let mut best_idx = 0;
    let mut best_val = fitness[0];
    for (i, &f) in fitness.iter().enumerate() {
        if f < best_val {
            best_val = f;
            best_idx = i;
        }
    }
    (best_idx, best_val)
}

fn compute_fitness_range(fitness: &[f64]) -> f64 {
    let max = fitness.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = fitness.iter().cloned().fold(f64::INFINITY, f64::min);
    max - min
}

fn validate_bounds<R: Runtime>(lower: &Tensor<R>, upper: &Tensor<R>) -> OptimizeResult<()> {
    let lower_data: Vec<f64> = lower.to_vec();
    let upper_data: Vec<f64> = upper.to_vec();

    for (i, (&l, &u)) in lower_data.iter().zip(upper_data.iter()).enumerate() {
        if l >= u {
            return Err(OptimizeError::InvalidInterval {
                a: l,
                b: u,
                context: format!("de: invalid bounds for dimension {}", i),
            });
        }
    }
    Ok(())
}

fn clamp_to_bounds<R, C>(
    client: &C,
    x: &Tensor<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R>,
{
    let clamped_upper = client
        .minimum(x, upper)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: min with upper - {}", e),
        })?;
    client
        .maximum(&clamped_upper, lower)
        .map_err(|e| OptimizeError::NumericalError {
            message: format!("de: max with lower - {}", e),
        })
}
