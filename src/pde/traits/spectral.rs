//! Spectral method traits for PDE solving.
use crate::DType;

use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::pde::error::PdeResult;
use crate::pde::types::{BoundarySpec, SpectralResult};

/// Spectral (Chebyshev collocation) algorithms for PDE solving.
pub trait SpectralAlgorithms<R: Runtime<DType = DType>> {
    /// 1D spectral: solve -u'' + q(x)u = f(x) on [-1,1] with Chebyshev collocation.
    ///
    /// Builds Chebyshev differentiation matrix D, solves (D^2 - diag(q))u = f
    /// as a dense linear system.
    fn spectral_1d(
        &self,
        f_rhs: &Tensor<R>,
        q: Option<&Tensor<R>>,
        n: usize,
        boundary: &[BoundarySpec<R>],
    ) -> PdeResult<SpectralResult<R>>;

    /// 2D spectral: solve -nabla^2 u = f on [-1,1]^2 with Chebyshev collocation.
    ///
    /// Uses tensor product: D_xx = D^2 kron I, D_yy = I kron D^2 via kron.
    fn spectral_2d(
        &self,
        f_rhs: &Tensor<R>,
        nx: usize,
        ny: usize,
        boundary: &[BoundarySpec<R>],
    ) -> PdeResult<SpectralResult<R>>;
}
