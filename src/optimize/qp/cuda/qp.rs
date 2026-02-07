//! CUDA implementation of QP solver.

use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::qp::impl_generic::{active_set_qp_impl, interior_point_qp_impl};
use crate::optimize::qp::traits::{QpAlgorithms, QpMethod, QpOptions, QpResult};

impl QpAlgorithms<CudaRuntime> for CudaClient {
    fn solve_qp(
        &self,
        q: &Tensor<CudaRuntime>,
        c: &Tensor<CudaRuntime>,
        a_eq: Option<&Tensor<CudaRuntime>>,
        b_eq: Option<&Tensor<CudaRuntime>>,
        a_ineq: Option<&Tensor<CudaRuntime>>,
        b_ineq: Option<&Tensor<CudaRuntime>>,
        options: &QpOptions,
    ) -> OptimizeResult<QpResult<CudaRuntime>> {
        match options.method {
            QpMethod::ActiveSet => {
                active_set_qp_impl(self, q, c, a_eq, b_eq, a_ineq, b_ineq, options)
            }
            QpMethod::InteriorPoint => {
                interior_point_qp_impl(self, q, c, a_eq, b_eq, a_ineq, b_ineq, options)
            }
        }
    }
}
