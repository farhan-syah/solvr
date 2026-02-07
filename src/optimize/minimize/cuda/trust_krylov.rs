//! CUDA implementation of trust-Krylov optimization.

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::minimize::impl_generic::trust_krylov_impl;
use crate::optimize::minimize::traits::trust_krylov::TrustKrylovAlgorithms;
use crate::optimize::minimize::traits::trust_region::{TrustRegionOptions, TrustRegionResult};

impl TrustKrylovAlgorithms<CudaRuntime> for CudaClient {
    fn trust_krylov<F>(
        &self,
        f: F,
        x0: &Tensor<CudaRuntime>,
        options: &TrustRegionOptions,
    ) -> OptimizeResult<TrustRegionResult<CudaRuntime>>
    where
        F: Fn(&Var<CudaRuntime>, &Self) -> NumrResult<Var<CudaRuntime>>,
    {
        trust_krylov_impl(self, f, x0, options)
    }
}
