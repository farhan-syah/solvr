//! Global optimization algorithms.
//!
//! Provides methods for finding global minima of functions,
//! avoiding local minima traps that affect local optimization methods.

mod cpu;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "wgpu")]
mod wgpu;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

// Re-export CPU convenience functions
pub use cpu::{basinhopping, differential_evolution, dual_annealing, simulated_annealing};

/// Options for global optimization.
#[derive(Debug, Clone)]
pub struct GlobalOptions {
    /// Maximum number of iterations/generations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: f64,
    /// Random seed (None for random)
    pub seed: Option<u64>,
}

impl Default for GlobalOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-8,
            seed: None,
        }
    }
}

/// Result from a global optimization method (scalar API).
#[derive(Debug, Clone)]
pub struct GlobalResult {
    pub x: Vec<f64>,
    pub fun: f64,
    pub iterations: usize,
    pub nfev: usize,
    pub converged: bool,
}

/// Algorithmic contract for global optimization operations.
pub trait GlobalOptimizationAlgorithms<R: Runtime> {
    /// Simulated annealing global optimizer.
    fn simulated_annealing<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<R>,
        upper_bounds: &Tensor<R>,
        options: &GlobalOptions,
    ) -> Result<GlobalTensorResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;

    /// Differential Evolution global optimizer.
    fn differential_evolution<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<R>,
        upper_bounds: &Tensor<R>,
        options: &GlobalOptions,
    ) -> Result<GlobalTensorResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;
}

/// Result from tensor-based global optimization.
#[derive(Debug, Clone)]
pub struct GlobalTensorResult<R: Runtime> {
    pub x: Tensor<R>,
    pub fun: f64,
    pub iterations: usize,
    pub nfev: usize,
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }

    #[test]
    fn test_compare_methods() {
        let bounds = vec![(-5.0, 5.0), (-5.0, 5.0)];
        let opts = GlobalOptions {
            max_iter: 500,
            seed: Some(42),
            ..Default::default()
        };

        let de_result = differential_evolution(sphere, &bounds, &opts).expect("DE failed");
        let sa_result = simulated_annealing(sphere, &bounds, &opts).expect("SA failed");
        let da_result = dual_annealing(sphere, &bounds, &opts).expect("DA failed");

        assert!(de_result.fun < 1e-4);
        assert!(sa_result.fun < 1.0); // SA is stochastic, use relaxed tolerance
        assert!(da_result.fun < 1e-4);
    }
}
