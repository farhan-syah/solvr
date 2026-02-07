//! CUDA implementation of trust-NCG optimization.

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::minimize::impl_generic::trust_ncg_impl;
use crate::optimize::minimize::traits::trust_ncg::TrustNcgAlgorithms;
use crate::optimize::minimize::traits::trust_region::{TrustRegionOptions, TrustRegionResult};

impl TrustNcgAlgorithms<CudaRuntime> for CudaClient {
    fn trust_ncg<F>(
        &self,
        f: F,
        x0: &Tensor<CudaRuntime>,
        options: &TrustRegionOptions,
    ) -> OptimizeResult<TrustRegionResult<CudaRuntime>>
    where
        F: Fn(&Var<CudaRuntime>, &Self) -> NumrResult<Var<CudaRuntime>>,
    {
        trust_ncg_impl(self, f, x0, options)
    }
}
