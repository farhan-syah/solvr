//! CUDA implementation of SDP.

use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::optimize::conic::impl_generic::sdp_impl;
use crate::optimize::conic::traits::sdp::{SdpAlgorithms, SdpOptions, SdpResult};
use crate::optimize::error::OptimizeResult;

impl SdpAlgorithms<CudaRuntime> for CudaClient {
    fn solve_sdp(
        &self,
        c_mat: &Tensor<CudaRuntime>,
        a_mats: &[Tensor<CudaRuntime>],
        b_vec: &Tensor<CudaRuntime>,
        options: &SdpOptions,
    ) -> OptimizeResult<SdpResult<CudaRuntime>> {
        sdp_impl(self, c_mat, a_mats, b_vec, options)
    }
}
