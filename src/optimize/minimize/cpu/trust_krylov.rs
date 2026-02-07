//! CPU implementation of trust-Krylov optimization.

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::minimize::impl_generic::trust_krylov_impl;
use crate::optimize::minimize::traits::trust_krylov::TrustKrylovAlgorithms;
use crate::optimize::minimize::traits::trust_region::{TrustRegionOptions, TrustRegionResult};

impl TrustKrylovAlgorithms<CpuRuntime> for CpuClient {
    fn trust_krylov<F>(
        &self,
        f: F,
        x0: &Tensor<CpuRuntime>,
        options: &TrustRegionOptions,
    ) -> OptimizeResult<TrustRegionResult<CpuRuntime>>
    where
        F: Fn(&Var<CpuRuntime>, &Self) -> NumrResult<Var<CpuRuntime>>,
    {
        trust_krylov_impl(self, f, x0, options)
    }
}
