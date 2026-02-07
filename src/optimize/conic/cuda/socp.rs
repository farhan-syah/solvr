//! CUDA implementation of SOCP.

use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::optimize::conic::impl_generic::socp_impl;
use crate::optimize::conic::traits::socp::{
    SocConstraint, SocpAlgorithms, SocpOptions, SocpResult,
};
use crate::optimize::error::OptimizeResult;

impl SocpAlgorithms<CudaRuntime> for CudaClient {
    fn solve_socp(
        &self,
        c: &Tensor<CudaRuntime>,
        constraints: &[SocConstraint<CudaRuntime>],
        options: &SocpOptions,
    ) -> OptimizeResult<SocpResult<CudaRuntime>> {
        socp_impl(self, c, constraints, options)
    }
}
