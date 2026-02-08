//! CUDA implementation of spectral PDE solvers.

use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::pde::error::PdeResult;
use crate::pde::impl_generic::{spectral_1d_impl, spectral_2d_impl};
use crate::pde::traits::SpectralAlgorithms;
use crate::pde::types::{BoundarySpec, SpectralResult};

impl SpectralAlgorithms<CudaRuntime> for CudaClient {
    fn spectral_1d(
        &self,
        f_rhs: &Tensor<CudaRuntime>,
        q: Option<&Tensor<CudaRuntime>>,
        n: usize,
        boundary: &[BoundarySpec<CudaRuntime>],
    ) -> PdeResult<SpectralResult<CudaRuntime>> {
        spectral_1d_impl(self, f_rhs, q, n, boundary)
    }

    fn spectral_2d(
        &self,
        f_rhs: &Tensor<CudaRuntime>,
        nx: usize,
        ny: usize,
        boundary: &[BoundarySpec<CudaRuntime>],
    ) -> PdeResult<SpectralResult<CudaRuntime>> {
        spectral_2d_impl(self, f_rhs, nx, ny, boundary)
    }
}
