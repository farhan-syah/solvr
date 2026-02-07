//! WebGPU implementation of trust-NCG optimization.

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::minimize::impl_generic::trust_ncg_impl;
use crate::optimize::minimize::traits::trust_ncg::TrustNcgAlgorithms;
use crate::optimize::minimize::traits::trust_region::{TrustRegionOptions, TrustRegionResult};

impl TrustNcgAlgorithms<WgpuRuntime> for WgpuClient {
    fn trust_ncg<F>(
        &self,
        f: F,
        x0: &Tensor<WgpuRuntime>,
        options: &TrustRegionOptions,
    ) -> OptimizeResult<TrustRegionResult<WgpuRuntime>>
    where
        F: Fn(&Var<WgpuRuntime>, &Self) -> NumrResult<Var<WgpuRuntime>>,
    {
        trust_ncg_impl(self, f, x0, options)
    }
}
