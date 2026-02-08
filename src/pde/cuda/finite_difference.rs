//! CUDA implementation of finite difference PDE solvers.

use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::pde::error::PdeResult;
use crate::pde::impl_generic::{heat_2d_impl, heat_3d_impl, poisson_impl, wave_impl};
use crate::pde::traits::FiniteDifferenceAlgorithms;
use crate::pde::types::{
    BoundarySpec, FdmOptions, FdmResult, Grid2D, Grid3D, TimeDependentOptions, TimeResult,
};

impl FiniteDifferenceAlgorithms<CudaRuntime> for CudaClient {
    fn fdm_poisson(
        &self,
        f: &Tensor<CudaRuntime>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<CudaRuntime>],
        options: &FdmOptions,
    ) -> PdeResult<FdmResult<CudaRuntime>> {
        poisson_impl(self, f, grid, boundary, options)
    }

    fn fdm_heat_2d(
        &self,
        u0: &Tensor<CudaRuntime>,
        alpha: f64,
        source: Option<&Tensor<CudaRuntime>>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<CudaRuntime>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<CudaRuntime>> {
        heat_2d_impl(self, u0, alpha, source, grid, boundary, time_opts, options)
    }

    fn fdm_heat_3d(
        &self,
        u0: &Tensor<CudaRuntime>,
        alpha: f64,
        source: Option<&Tensor<CudaRuntime>>,
        grid: &Grid3D,
        boundary: &[BoundarySpec<CudaRuntime>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<CudaRuntime>> {
        heat_3d_impl(self, u0, alpha, source, grid, boundary, time_opts, options)
    }

    fn fdm_wave(
        &self,
        u0: &Tensor<CudaRuntime>,
        v0: &Tensor<CudaRuntime>,
        c: f64,
        source: Option<&Tensor<CudaRuntime>>,
        grid: &Grid2D,
        boundary: &[BoundarySpec<CudaRuntime>],
        time_opts: &TimeDependentOptions,
        options: &FdmOptions,
    ) -> PdeResult<TimeResult<CudaRuntime>> {
        wave_impl(self, u0, v0, c, source, grid, boundary, time_opts, options)
    }
}
