//! Global optimization algorithms.
//!
//! Provides methods for finding global minima of functions,
//! avoiding local minima traps that affect local optimization methods.
//!
//! All algorithms use tensor operations and are generic over `R: Runtime`.

mod cpu;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "wgpu")]
mod wgpu;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

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

/// Result from tensor-based global optimization.
#[derive(Debug, Clone)]
pub struct GlobalTensorResult<R: Runtime> {
    pub x: Tensor<R>,
    pub fun: f64,
    pub iterations: usize,
    pub nfev: usize,
    pub converged: bool,
}

/// Algorithmic contract for global optimization operations.
pub trait GlobalOptimizationAlgorithms<R: Runtime> {
    /// Simulated annealing global optimizer.
    ///
    /// Uses probabilistic acceptance of worse solutions to escape local minima.
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
    ///
    /// Population-based optimizer using mutation, crossover, and selection.
    fn differential_evolution<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<R>,
        upper_bounds: &Tensor<R>,
        options: &GlobalOptions,
    ) -> Result<GlobalTensorResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;

    /// Basin-hopping global optimizer.
    ///
    /// Combines local minimization with random perturbations to escape local minima.
    fn basinhopping<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        lower_bounds: &Tensor<R>,
        upper_bounds: &Tensor<R>,
        options: &GlobalOptions,
    ) -> Result<GlobalTensorResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;

    /// Dual annealing global optimizer.
    ///
    /// Combines simulated annealing with local search for smooth functions.
    fn dual_annealing<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<R>,
        upper_bounds: &Tensor<R>,
        options: &GlobalOptions,
    ) -> Result<GlobalTensorResult<R>>
    where
        F: Fn(&Tensor<R>) -> Result<f64>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn sphere_tensor<R: Runtime>(x: &Tensor<R>) -> Result<f64> {
        let data: Vec<f64> = x.to_vec();
        Ok(data.iter().map(|&xi| xi * xi).sum())
    }

    #[test]
    fn test_compare_methods() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let lower = Tensor::<CpuRuntime>::from_slice(&[-5.0, -5.0], &[2], &device);
        let upper = Tensor::<CpuRuntime>::from_slice(&[5.0, 5.0], &[2], &device);

        // DE and DA converge faster
        let opts = GlobalOptions {
            max_iter: 500,
            seed: Some(42),
            ..Default::default()
        };

        let de_result = client
            .differential_evolution(sphere_tensor, &lower, &upper, &opts)
            .expect("DE failed");
        let da_result = client
            .dual_annealing(sphere_tensor, &lower, &upper, &opts)
            .expect("DA failed");

        assert!(de_result.fun < 1e-4);
        assert!(da_result.fun < 1e-4);

        // SA needs more iterations for similar convergence
        let sa_opts = GlobalOptions {
            max_iter: 5000,
            seed: Some(42),
            ..Default::default()
        };

        let sa_result = client
            .simulated_annealing(sphere_tensor, &lower, &upper, &sa_opts)
            .expect("SA failed");
        assert!(sa_result.fun < 1.0); // SA is stochastic, use relaxed tolerance
    }

    #[test]
    fn test_basinhopping() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let x0 = Tensor::<CpuRuntime>::from_slice(&[2.0, 2.0, 2.0], &[3], &device);
        let lower = Tensor::<CpuRuntime>::from_slice(&[-10.0, -10.0, -10.0], &[3], &device);
        let upper = Tensor::<CpuRuntime>::from_slice(&[10.0, 10.0, 10.0], &[3], &device);

        let opts = GlobalOptions {
            max_iter: 50,
            seed: Some(42),
            ..Default::default()
        };

        let result = client
            .basinhopping(sphere_tensor, &x0, &lower, &upper, &opts)
            .expect("basinhopping failed");
        assert!(result.fun < 1e-4);
    }
}
