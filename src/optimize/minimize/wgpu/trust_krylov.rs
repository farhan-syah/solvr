//! WebGPU implementation of trust-Krylov optimization.

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::minimize::impl_generic::trust_krylov_impl;
use crate::optimize::minimize::traits::trust_krylov::TrustKrylovAlgorithms;
use crate::optimize::minimize::traits::trust_region::{TrustRegionOptions, TrustRegionResult};

impl TrustKrylovAlgorithms<WgpuRuntime> for WgpuClient {
    fn trust_krylov<F>(
        &self,
        f: F,
        x0: &Tensor<WgpuRuntime>,
        options: &TrustRegionOptions,
    ) -> OptimizeResult<TrustRegionResult<WgpuRuntime>>
    where
        F: Fn(&Var<WgpuRuntime>, &Self) -> NumrResult<Var<WgpuRuntime>>,
    {
        trust_krylov_impl(self, f, x0, options)
    }
}
