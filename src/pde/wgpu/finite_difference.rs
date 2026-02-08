//! WebGPU implementation of finite difference PDE solvers.

use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::pde::error::PdeResult;
use crate::pde::impl_generic::{heat_2d_impl, heat_3d_impl, poisson_impl, wave_impl};
use crate::pde::traits::FiniteDifferenceAlgorithms;
use crate::pde::types::{
    BoundarySpec, FdmOptions, FdmResult, Grid2D, Grid3D, TimeDependentOptions, TimeResult,
};

impl FiniteDifferenceAlgorithms<WgpuRuntime> for WgpuClient {
    fn fdm_poisson(
        &self,
        f: &Tensor<WgpuRuntime>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<WgpuRuntime>],
        options: &FdmOptions,
    ) -> PdeResult<FdmResult<WgpuRuntime>> {
        poisson_impl(self, f, grid, boundary, options)
    }

    fn fdm_heat_2d(
        &self,
        u0: &Tensor<WgpuRuntime>,
        alpha: f64,
        source: Option<&Tensor<WgpuRuntime>>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<WgpuRuntime>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<WgpuRuntime>> {
        heat_2d_impl(self, u0, alpha, source, grid, boundary, time_opts, options)
    }

    fn fdm_heat_3d(
        &self,
        u0: &Tensor<WgpuRuntime>,
        alpha: f64,
        source: Option<&Tensor<WgpuRuntime>>,
        grid: &Grid3D,
        boundary: &[BoundarySpec<WgpuRuntime>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<WgpuRuntime>> {
        heat_3d_impl(self, u0, alpha, source, grid, boundary, time_opts, options)
    }

    fn fdm_wave(
        &self,
        u0: &Tensor<WgpuRuntime>,
        v0: &Tensor<WgpuRuntime>,
        c: f64,
        source: Option<&Tensor<WgpuRuntime>>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<WgpuRuntime>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<WgpuRuntime>> {
        wave_impl(self, u0, v0, c, source, grid, boundary, time_opts, options)
    }
}
