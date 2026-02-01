//! CPU implementation of global optimization algorithms.
//!
//! This module implements the [`GlobalOptimizationAlgorithms`] trait for CPU
//! and provides scalar convenience functions.

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::global::{
    GlobalOptimizationAlgorithms, GlobalOptions, GlobalResult, GlobalTensorResult,
};
use crate::optimize::impl_generic::global::{
    differential_evolution_impl, simulated_annealing_impl,
};
use crate::optimize::impl_generic::nelder_mead_impl;
use crate::optimize::minimize::MinimizeOptions;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Trait Implementation
// ============================================================================

impl GlobalOptimizationAlgorithms<CpuRuntime> for CpuClient {
    fn simulated_annealing<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<CpuRuntime>,
        upper_bounds: &Tensor<CpuRuntime>,
        options: &GlobalOptions,
    ) -> Result<GlobalTensorResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<f64>,
    {
        let result = simulated_annealing_impl(self, f, lower_bounds, upper_bounds, options)
            .map_err(|e| {
                numr::error::Error::backend_limitation("cpu", "simulated_annealing", e.to_string())
            })?;
        Ok(GlobalTensorResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            nfev: result.nfev,
            converged: result.converged,
        })
    }

    fn differential_evolution<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<CpuRuntime>,
        upper_bounds: &Tensor<CpuRuntime>,
        options: &GlobalOptions,
    ) -> Result<GlobalTensorResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<f64>,
    {
        let result = differential_evolution_impl(self, f, lower_bounds, upper_bounds, options)
            .map_err(|e| {
                numr::error::Error::backend_limitation(
                    "cpu",
                    "differential_evolution",
                    e.to_string(),
                )
            })?;
        Ok(GlobalTensorResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            nfev: result.nfev,
            converged: result.converged,
        })
    }
}

// ============================================================================
// Scalar Convenience Functions
// ============================================================================

/// Simulated annealing global optimizer (scalar API).
pub fn simulated_annealing<F>(
    f: F,
    bounds: &[(f64, f64)],
    options: &GlobalOptions,
) -> OptimizeResult<GlobalResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n = bounds.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "simulated_annealing: empty bounds".to_string(),
        });
    }

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let lower: Vec<f64> = bounds.iter().map(|&(l, _)| l).collect();
    let upper: Vec<f64> = bounds.iter().map(|&(_, u)| u).collect();

    let lower_tensor = Tensor::<CpuRuntime>::from_slice(&lower, &[n], &device);
    let upper_tensor = Tensor::<CpuRuntime>::from_slice(&upper, &[n], &device);

    let tensor_f = |x: &Tensor<CpuRuntime>| -> Result<f64> {
        let x_data: Vec<f64> = x.to_vec();
        Ok(f(&x_data))
    };

    let result = client
        .simulated_annealing(tensor_f, &lower_tensor, &upper_tensor, options)
        .map_err(|e| OptimizeError::NumericalError {
            message: e.to_string(),
        })?;

    Ok(GlobalResult {
        x: result.x.to_vec(),
        fun: result.fun,
        iterations: result.iterations,
        nfev: result.nfev,
        converged: result.converged,
    })
}

/// Differential Evolution global optimizer (scalar API).
pub fn differential_evolution<F>(
    f: F,
    bounds: &[(f64, f64)],
    options: &GlobalOptions,
) -> OptimizeResult<GlobalResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n = bounds.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "differential_evolution: empty bounds".to_string(),
        });
    }

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let lower: Vec<f64> = bounds.iter().map(|&(l, _)| l).collect();
    let upper: Vec<f64> = bounds.iter().map(|&(_, u)| u).collect();

    let lower_tensor = Tensor::<CpuRuntime>::from_slice(&lower, &[n], &device);
    let upper_tensor = Tensor::<CpuRuntime>::from_slice(&upper, &[n], &device);

    let tensor_f = |x: &Tensor<CpuRuntime>| -> Result<f64> {
        let x_data: Vec<f64> = x.to_vec();
        Ok(f(&x_data))
    };

    let result = client
        .differential_evolution(tensor_f, &lower_tensor, &upper_tensor, options)
        .map_err(|e| OptimizeError::NumericalError {
            message: e.to_string(),
        })?;

    Ok(GlobalResult {
        x: result.x.to_vec(),
        fun: result.fun,
        iterations: result.iterations,
        nfev: result.nfev,
        converged: result.converged,
    })
}

/// Basin-hopping global optimizer (scalar API).
///
/// Combines local minimization with random perturbations to escape local minima.
pub fn basinhopping<F>(f: F, x0: &[f64], options: &GlobalOptions) -> OptimizeResult<GlobalResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n = x0.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "basinhopping: empty initial guess".to_string(),
        });
    }

    let mut rng = SimpleRng::new(options.seed.unwrap_or(12345));
    let step_size = 0.5;
    let temperature = 1.0;

    let local_opts = MinimizeOptions {
        max_iter: 100,
        f_tol: 1e-6,
        x_tol: 1e-6,
        g_tol: 1e-6,
        eps: 1e-8,
    };

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let tensor_f = |x: &Tensor<CpuRuntime>| -> Result<f64> {
        let data: Vec<f64> = x.to_vec();
        Ok(f(&data))
    };

    let x0_tensor = Tensor::<CpuRuntime>::from_slice(x0, &[n], &device);
    let local_result = nelder_mead_impl(&client, tensor_f, &x0_tensor, &local_opts)?;
    let mut x_current: Vec<f64> = local_result.x.to_vec();
    let mut f_current = local_result.fun;
    let mut nfev = local_result.nfev;

    let mut x_best = x_current.clone();
    let mut f_best = f_current;

    for iter in 0..options.max_iter {
        let x_perturbed: Vec<f64> = x_current
            .iter()
            .map(|&xi| xi + step_size * (2.0 * rng.next_f64() - 1.0))
            .collect();

        let x_perturbed_tensor = Tensor::<CpuRuntime>::from_slice(&x_perturbed, &[n], &device);
        let local_result = nelder_mead_impl(&client, tensor_f, &x_perturbed_tensor, &local_opts)?;
        let x_new: Vec<f64> = local_result.x.to_vec();
        let f_new = local_result.fun;
        nfev += local_result.nfev;

        let delta_f = f_new - f_current;
        let accept = if delta_f < 0.0 {
            true
        } else {
            rng.next_f64() < (-delta_f / temperature).exp()
        };

        if accept {
            x_current = x_new;
            f_current = f_new;

            if f_current < f_best {
                x_best = x_current.clone();
                f_best = f_current;
            }
        }

        if (f_current - f_best).abs() < options.tol && iter > 10 {
            return Ok(GlobalResult {
                x: x_best,
                fun: f_best,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }
    }

    Ok(GlobalResult {
        x: x_best,
        fun: f_best,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

/// Dual annealing global optimizer (scalar API).
///
/// Combines simulated annealing with local search for smooth functions.
pub fn dual_annealing<F>(
    f: F,
    bounds: &[(f64, f64)],
    options: &GlobalOptions,
) -> OptimizeResult<GlobalResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n = bounds.len();
    if n == 0 {
        return Err(OptimizeError::InvalidInput {
            context: "dual_annealing: empty bounds".to_string(),
        });
    }

    for (i, &(lo, hi)) in bounds.iter().enumerate() {
        if lo >= hi {
            return Err(OptimizeError::InvalidInterval {
                a: lo,
                b: hi,
                context: format!("dual_annealing: invalid bounds for dimension {}", i),
            });
        }
    }

    let mut rng = SimpleRng::new(options.seed.unwrap_or(12345));

    let local_opts = MinimizeOptions {
        max_iter: 50,
        f_tol: 1e-6,
        x_tol: 1e-6,
        g_tol: 1e-6,
        eps: 1e-8,
    };

    let device = CpuDevice::new();
    let client = CpuClient::new(device.clone());

    let tensor_f = |x: &Tensor<CpuRuntime>| -> Result<f64> {
        let data: Vec<f64> = x.to_vec();
        Ok(f(&data))
    };

    let t_initial: f64 = 5230.0;
    let t_final: f64 = 0.001;
    let cooling_rate = (t_final / t_initial).powf(1.0 / (options.max_iter as f64 / 2.0));
    let local_search_interval = 10;

    let mut x_current: Vec<f64> = bounds
        .iter()
        .map(|&(lo, hi)| rng.next_range(lo, hi))
        .collect();
    let mut f_current = f(&x_current);
    let mut nfev = 1;

    let mut x_best = x_current.clone();
    let mut f_best = f_current;
    let mut temperature = t_initial;

    for iter in 0..options.max_iter {
        // Occasional local search
        if iter > 0 && iter % local_search_interval == 0 {
            let x_tensor = Tensor::<CpuRuntime>::from_slice(&x_current, &[n], &device);
            let local_result = nelder_mead_impl(&client, tensor_f, &x_tensor, &local_opts)?;
            nfev += local_result.nfev;

            let x_local_raw: Vec<f64> = local_result.x.to_vec();
            let x_local: Vec<f64> = x_local_raw
                .iter()
                .enumerate()
                .map(|(i, &xi)| xi.clamp(bounds[i].0, bounds[i].1))
                .collect();
            let f_local = f(&x_local);
            nfev += 1;

            if f_local < f_current {
                x_current = x_local;
                f_current = f_local;

                if f_current < f_best {
                    x_best = x_current.clone();
                    f_best = f_current;
                }
            }
        }

        // Generalized simulated annealing step (Cauchy visiting distribution)
        let scale = temperature / t_initial;
        let x_neighbor: Vec<f64> = x_current
            .iter()
            .enumerate()
            .map(|(i, &xi)| {
                let range = bounds[i].1 - bounds[i].0;
                let u = rng.next_f64();
                let cauchy = scale * range * (std::f64::consts::PI * (u - 0.5)).tan();
                (xi + cauchy * 0.1).clamp(bounds[i].0, bounds[i].1)
            })
            .collect();

        let f_neighbor = f(&x_neighbor);
        nfev += 1;

        let delta = f_neighbor - f_current;
        let accept = if delta < 0.0 {
            true
        } else {
            rng.next_f64() < (-delta / temperature).exp()
        };

        if accept {
            x_current = x_neighbor;
            f_current = f_neighbor;

            if f_current < f_best {
                x_best = x_current.clone();
                f_best = f_current;
            }
        }

        temperature *= cooling_rate;

        if temperature < t_final || (f_best < options.tol && iter > 100) {
            // Final local search
            let x_tensor = Tensor::<CpuRuntime>::from_slice(&x_best, &[n], &device);
            let local_result = nelder_mead_impl(&client, tensor_f, &x_tensor, &local_opts)?;
            nfev += local_result.nfev;

            let x_final_raw: Vec<f64> = local_result.x.to_vec();
            let x_final: Vec<f64> = x_final_raw
                .iter()
                .enumerate()
                .map(|(i, &xi)| xi.clamp(bounds[i].0, bounds[i].1))
                .collect();
            let f_final = f(&x_final);
            nfev += 1;

            if f_final < f_best {
                x_best = x_final;
                f_best = f_final;
            }

            return Ok(GlobalResult {
                x: x_best,
                fun: f_best,
                iterations: iter + 1,
                nfev,
                converged: true,
            });
        }
    }

    Ok(GlobalResult {
        x: x_best,
        fun: f_best,
        iterations: options.max_iter,
        nfev,
        converged: false,
    })
}

// ============================================================================
// Helper: Simple RNG for reproducibility
// ============================================================================

/// Simple linear congruential generator for reproducible randomness.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    fn rosenbrock(x: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..x.len() - 1 {
            sum += 100.0 * (x[i + 1] - x[i] * x[i]).powi(2) + (1.0 - x[i]).powi(2);
        }
        sum
    }

    fn ackley(x: &[f64]) -> f64 {
        let n = x.len() as f64;
        let sum_sq: f64 = x.iter().map(|&xi| xi * xi).sum();
        let sum_cos: f64 = x
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
            .sum();
        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
            + 20.0
            + std::f64::consts::E
    }

    #[test]
    fn test_simulated_annealing_sphere() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = GlobalOptions {
            max_iter: 10000,
            seed: Some(42),
            ..Default::default()
        };

        let result = simulated_annealing(sphere, &bounds, &opts).expect("SA failed");
        assert!(result.fun < 1.0);
    }

    #[test]
    fn test_de_sphere() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = GlobalOptions {
            seed: Some(42),
            ..Default::default()
        };

        let result = differential_evolution(sphere, &bounds, &opts).expect("DE failed");
        assert!(result.fun < 1e-6);
    }

    #[test]
    fn test_de_empty_bounds() {
        let result = differential_evolution(sphere, &[], &GlobalOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInput { .. })));
    }

    #[test]
    fn test_basinhopping_sphere() {
        let opts = GlobalOptions {
            max_iter: 50,
            seed: Some(42),
            ..Default::default()
        };

        let result = basinhopping(sphere, &[2.0, 2.0, 2.0], &opts).expect("basinhopping failed");
        assert!(result.fun < 1e-4);
    }

    #[test]
    fn test_basinhopping_rosenbrock() {
        let opts = GlobalOptions {
            max_iter: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = basinhopping(rosenbrock, &[0.0, 0.0], &opts).expect("basinhopping failed");
        assert!(result.fun < 0.1);
    }

    #[test]
    fn test_dual_annealing_sphere() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = GlobalOptions {
            max_iter: 500,
            seed: Some(42),
            ..Default::default()
        };

        let result = dual_annealing(sphere, &bounds, &opts).expect("DA failed");
        assert!(result.fun < 1e-4);
    }

    #[test]
    fn test_dual_annealing_ackley() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = GlobalOptions {
            max_iter: 1000,
            seed: Some(42),
            ..Default::default()
        };

        let result = dual_annealing(ackley, &bounds, &opts).expect("DA failed");
        assert!(result.fun < 3.0);
    }
}
