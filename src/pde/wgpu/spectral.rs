//! WebGPU implementation of spectral PDE solvers.

use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::pde::error::PdeResult;
use crate::pde::impl_generic::{spectral_1d_impl, spectral_2d_impl};
use crate::pde::traits::SpectralAlgorithms;
use crate::pde::types::{BoundarySpec, SpectralResult};

impl SpectralAlgorithms<WgpuRuntime> for WgpuClient {
    fn spectral_1d(
        &self,
        f_rhs: &Tensor<WgpuRuntime>,
        q: Option<&Tensor<WgpuRuntime>>,
        n: usize,
        boundary: &[BoundarySpec<WgpuRuntime>],
    ) -> PdeResult<SpectralResult<WgpuRuntime>> {
        spectral_1d_impl(self, f_rhs, q, n, boundary)
    }

    fn spectral_2d(
        &self,
        f_rhs: &Tensor<WgpuRuntime>,
        nx: usize,
        ny: usize,
        boundary: &[BoundarySpec<WgpuRuntime>],
    ) -> PdeResult<SpectralResult<WgpuRuntime>> {
        spectral_2d_impl(self, f_rhs, nx, ny, boundary)
    }
}
