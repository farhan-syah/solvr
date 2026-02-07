//! WebGPU implementation of SDP.

use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::conic::impl_generic::sdp_impl;
use crate::optimize::conic::traits::sdp::{SdpAlgorithms, SdpOptions, SdpResult};
use crate::optimize::error::OptimizeResult;

impl SdpAlgorithms<WgpuRuntime> for WgpuClient {
    fn solve_sdp(
        &self,
        c_mat: &Tensor<WgpuRuntime>,
        a_mats: &[Tensor<WgpuRuntime>],
        b_vec: &Tensor<WgpuRuntime>,
        options: &SdpOptions,
    ) -> OptimizeResult<SdpResult<WgpuRuntime>> {
        sdp_impl(self, c_mat, a_mats, b_vec, options)
    }
}
