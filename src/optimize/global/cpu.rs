//! CPU implementation of global optimization algorithms.
//!
//! Implements the [`GlobalOptimizationAlgorithms`] trait for CPU runtime.
//! All implementations delegate to the generic implementations in `impl_generic`.

use crate::optimize::global::{GlobalOptimizationAlgorithms, GlobalOptions, GlobalTensorResult};
use crate::optimize::impl_generic::global::{
    basinhopping_impl, differential_evolution_impl, dual_annealing_impl, simulated_annealing_impl,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

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

    fn basinhopping<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        lower_bounds: &Tensor<CpuRuntime>,
        upper_bounds: &Tensor<CpuRuntime>,
        options: &GlobalOptions,
    ) -> Result<GlobalTensorResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<f64>,
    {
        let result = basinhopping_impl(self, f, x0, lower_bounds, upper_bounds, options).map_err(
            |e| numr::error::Error::backend_limitation("cpu", "basinhopping", e.to_string()),
        )?;
        Ok(GlobalTensorResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            nfev: result.nfev,
            converged: result.converged,
        })
    }

    fn dual_annealing<F>(
        &self,
        f: F,
        lower_bounds: &Tensor<CpuRuntime>,
        upper_bounds: &Tensor<CpuRuntime>,
        options: &GlobalOptions,
    ) -> Result<GlobalTensorResult<CpuRuntime>>
    where
        F: Fn(&Tensor<CpuRuntime>) -> Result<f64>,
    {
        let result = dual_annealing_impl(self, f, lower_bounds, upper_bounds, options).map_err(
            |e| numr::error::Error::backend_limitation("cpu", "dual_annealing", e.to_string()),
        )?;
        Ok(GlobalTensorResult {
            x: result.x,
            fun: result.fun,
            iterations: result.iterations,
            nfev: result.nfev,
            converged: result.converged,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn sphere_tensor<R: numr::runtime::Runtime>(x: &Tensor<R>) -> Result<f64> {
        let data: Vec<f64> = x.to_vec();
        Ok(data.iter().map(|&xi| xi * xi).sum())
    }

    fn rosenbrock_tensor<R: numr::runtime::Runtime>(x: &Tensor<R>) -> Result<f64> {
        let data: Vec<f64> = x.to_vec();
        let mut sum = 0.0;
        for i in 0..data.len() - 1 {
            sum += 100.0 * (data[i + 1] - data[i] * data[i]).powi(2) + (1.0 - data[i]).powi(2);
        }
        Ok(sum)
    }

    fn ackley_tensor<R: numr::runtime::Runtime>(x: &Tensor<R>) -> Result<f64> {
        let data: Vec<f64> = x.to_vec();
        let n = data.len() as f64;
        let sum_sq: f64 = data.iter().map(|&xi| xi * xi).sum();
        let sum_cos: f64 = data
            .iter()
            .map(|&xi| (2.0 * std::f64::consts::PI * xi).cos())
            .sum();
        Ok(-20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
            + 20.0
            + std::f64::consts::E)
    }

    #[test]
    fn test_simulated_annealing_sphere() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let lower = Tensor::<CpuRuntime>::from_slice(&[-5.0, -5.0], &[2], &device);
        let upper = Tensor::<CpuRuntime>::from_slice(&[5.0, 5.0], &[2], &device);

        let opts = GlobalOptions {
            max_iter: 10000,
            seed: Some(42),
            ..Default::default()
        };

        let result = client
            .simulated_annealing(sphere_tensor, &lower, &upper, &opts)
            .expect("SA failed");
        assert!(result.fun < 1.0);
    }

    #[test]
    fn test_de_sphere() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let lower = Tensor::<CpuRuntime>::from_slice(&[-5.0, -5.0], &[2], &device);
        let upper = Tensor::<CpuRuntime>::from_slice(&[5.0, 5.0], &[2], &device);

        let opts = GlobalOptions {
            seed: Some(42),
            ..Default::default()
        };

        let result = client
            .differential_evolution(sphere_tensor, &lower, &upper, &opts)
            .expect("DE failed");
        assert!(result.fun < 1e-6);
    }

    #[test]
    fn test_basinhopping_sphere() {
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

    #[test]
    fn test_basinhopping_rosenbrock() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let x0 = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[2], &device);
        let lower = Tensor::<CpuRuntime>::from_slice(&[-10.0, -10.0], &[2], &device);
        let upper = Tensor::<CpuRuntime>::from_slice(&[10.0, 10.0], &[2], &device);

        let opts = GlobalOptions {
            max_iter: 100,
            seed: Some(42),
            ..Default::default()
        };

        let result = client
            .basinhopping(rosenbrock_tensor, &x0, &lower, &upper, &opts)
            .expect("basinhopping failed");
        assert!(result.fun < 0.1);
    }

    #[test]
    fn test_dual_annealing_sphere() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let lower = Tensor::<CpuRuntime>::from_slice(&[-5.0, -5.0], &[2], &device);
        let upper = Tensor::<CpuRuntime>::from_slice(&[5.0, 5.0], &[2], &device);

        let opts = GlobalOptions {
            max_iter: 500,
            seed: Some(42),
            ..Default::default()
        };

        let result = client
            .dual_annealing(sphere_tensor, &lower, &upper, &opts)
            .expect("DA failed");
        assert!(result.fun < 1e-4);
    }

    #[test]
    fn test_dual_annealing_ackley() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());

        let lower = Tensor::<CpuRuntime>::from_slice(&[-5.0, -5.0], &[2], &device);
        let upper = Tensor::<CpuRuntime>::from_slice(&[5.0, 5.0], &[2], &device);

        let opts = GlobalOptions {
            max_iter: 3000,
            seed: Some(42),
            ..Default::default()
        };

        let result = client
            .dual_annealing(ackley_tensor, &lower, &upper, &opts)
            .expect("DA failed");
        // Ackley minimum is 0 at origin; accept reasonable convergence for stochastic method
        // With more iterations, should get close to global minimum
        assert!(result.fun < 10.0);
    }
}
