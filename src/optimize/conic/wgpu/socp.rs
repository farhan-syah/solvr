//! WebGPU implementation of SOCP.

use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::conic::impl_generic::socp_impl;
use crate::optimize::conic::traits::socp::{
    SocConstraint, SocpAlgorithms, SocpOptions, SocpResult,
};
use crate::optimize::error::OptimizeResult;

impl SocpAlgorithms<WgpuRuntime> for WgpuClient {
    fn solve_socp(
        &self,
        c: &Tensor<WgpuRuntime>,
        constraints: &[SocConstraint<WgpuRuntime>],
        options: &SocpOptions,
    ) -> OptimizeResult<SocpResult<WgpuRuntime>> {
        socp_impl(self, c, constraints, options)
    }
}
