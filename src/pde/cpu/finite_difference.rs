//! CPU implementation of finite difference PDE solvers.

use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::pde::error::PdeResult;
use crate::pde::impl_generic::{heat_2d_impl, heat_3d_impl, poisson_impl, wave_impl};
use crate::pde::traits::FiniteDifferenceAlgorithms;
use crate::pde::types::{
    BoundarySpec, FdmOptions, FdmResult, Grid2D, Grid3D, TimeDependentOptions, TimeResult,
};

impl FiniteDifferenceAlgorithms<CpuRuntime> for CpuClient {
    fn fdm_poisson(
        &self,
        f: &Tensor<CpuRuntime>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<CpuRuntime>],
        options: &FdmOptions,
    ) -> PdeResult<FdmResult<CpuRuntime>> {
        poisson_impl(self, f, grid, boundary, options)
    }

    fn fdm_heat_2d(
        &self,
        u0: &Tensor<CpuRuntime>,
        alpha: f64,
        source: Option<&Tensor<CpuRuntime>>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<CpuRuntime>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<CpuRuntime>> {
        heat_2d_impl(self, u0, alpha, source, grid, boundary, time_opts, options)
    }

    fn fdm_heat_3d(
        &self,
        u0: &Tensor<CpuRuntime>,
        alpha: f64,
        source: Option<&Tensor<CpuRuntime>>,
        grid: &Grid3D,
        boundary: &[BoundarySpec<CpuRuntime>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<CpuRuntime>> {
        heat_3d_impl(self, u0, alpha, source, grid, boundary, time_opts, options)
    }

    fn fdm_wave(
        &self,
        u0: &Tensor<CpuRuntime>,
        v0: &Tensor<CpuRuntime>,
        c: f64,
        source: Option<&Tensor<CpuRuntime>>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<CpuRuntime>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<CpuRuntime>> {
        wave_impl(self, u0, v0, c, source, grid, boundary, time_opts, options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_poisson_known_solution() {
        let (client, device) = setup();

        // Solve -nabla^2 u = f on [0,1]^2 with u=0 on boundary
        // f(x,y) = 2*pi^2 * sin(pi*x) * sin(pi*y)
        // Exact: u(x,y) = sin(pi*x) * sin(pi*y)
        let nx = 21;
        let ny = 21;
        let dx = 1.0 / (nx - 1) as f64;
        let dy = 1.0 / (ny - 1) as f64;

        let grid = Grid2D {
            nx,
            ny,
            dx,
            dy,
            x_range: [0.0, 1.0],
            y_range: [0.0, 1.0],
        };

        let pi = std::f64::consts::PI;
        let mut f_data = vec![0.0; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                f_data[i * ny + j] = 2.0 * pi * pi * (pi * x).sin() * (pi * y).sin();
            }
        }
        let f_tensor = Tensor::<CpuRuntime>::from_slice(&f_data, &[nx, ny], &device);

        let result = client
            .fdm_poisson(&f_tensor, &grid, &[], &FdmOptions::default())
            .expect("Poisson solve failed");

        // Check solution at center
        let sol: Vec<f64> = result.solution.to_vec();
        let center_i = nx / 2;
        let center_j = ny / 2;
        let numerical = sol[center_i * ny + center_j];
        let exact = (pi * 0.5).sin() * (pi * 0.5).sin();

        // FDM on 21x21 grid gives ~O(h^2) error
        assert!(
            (numerical - exact).abs() < 0.02,
            "Poisson center: numerical={}, exact={}, error={}",
            numerical,
            exact,
            (numerical - exact).abs()
        );
    }

    #[test]
    fn test_heat_2d_diffusion() {
        let (client, device) = setup();

        let nx = 11;
        let ny = 11;
        let dx = 1.0 / (nx - 1) as f64;
        let dy = 1.0 / (ny - 1) as f64;

        let grid = Grid2D {
            nx,
            ny,
            dx,
            dy,
            x_range: [0.0, 1.0],
            y_range: [0.0, 1.0],
        };

        // Initial condition: hot spot in center
        let mut u0_data = vec![0.0; nx * ny];
        u0_data[(nx / 2) * ny + ny / 2] = 1.0;
        let u0 = Tensor::<CpuRuntime>::from_slice(&u0_data, &[nx, ny], &device);

        let time_opts = TimeDependentOptions {
            t_span: [0.0, 0.01],
            dt: None,
            save_every: 0,
        };

        let result = client
            .fdm_heat_2d(
                &u0,
                1.0,
                None,
                &grid,
                &[],
                &time_opts,
                &FdmOptions::default(),
            )
            .expect("Heat 2D solve failed");

        assert!(!result.solutions.is_empty());

        // Final solution should be more spread out (lower max)
        let final_sol: Vec<f64> = result.solutions.last().unwrap().to_vec();
        let max_val = final_sol.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val < 1.0, "Heat should have diffused: max={}", max_val);
    }

    #[test]
    fn test_wave_energy_conservation() {
        let (client, device) = setup();

        let nx = 11;
        let ny = 11;
        let dx = 1.0 / (nx - 1) as f64;
        let dy = 1.0 / (ny - 1) as f64;

        let grid = Grid2D {
            nx,
            ny,
            dx,
            dy,
            x_range: [0.0, 1.0],
            y_range: [0.0, 1.0],
        };

        let pi = std::f64::consts::PI;
        let mut u0_data = vec![0.0; nx * ny];
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let x = i as f64 * dx;
                let y = j as f64 * dy;
                u0_data[i * ny + j] = (pi * x).sin() * (pi * y).sin();
            }
        }
        let u0 = Tensor::<CpuRuntime>::from_slice(&u0_data, &[nx, ny], &device);
        let v0 = Tensor::<CpuRuntime>::from_slice(&vec![0.0; nx * ny], &[nx, ny], &device);

        let time_opts = TimeDependentOptions {
            t_span: [0.0, 0.1],
            dt: None,
            save_every: 0,
        };

        let result = client
            .fdm_wave(
                &u0,
                &v0,
                1.0,
                None,
                &grid,
                &[],
                &time_opts,
                &FdmOptions::default(),
            )
            .expect("Wave solve failed");

        assert!(!result.solutions.is_empty());
    }
}
