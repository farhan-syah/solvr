//! WebGPU implementation of QP solver.

use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

use crate::optimize::error::OptimizeResult;
use crate::optimize::qp::impl_generic::{active_set_qp_impl, interior_point_qp_impl};
use crate::optimize::qp::traits::{QpAlgorithms, QpMethod, QpOptions, QpResult};

impl QpAlgorithms<WgpuRuntime> for WgpuClient {
    fn solve_qp(
        &self,
        q: &Tensor<WgpuRuntime>,
        c: &Tensor<WgpuRuntime>,
        a_eq: Option<&Tensor<WgpuRuntime>>,
        b_eq: Option<&Tensor<WgpuRuntime>>,
        a_ineq: Option<&Tensor<WgpuRuntime>>,
        b_ineq: Option<&Tensor<WgpuRuntime>>,
        options: &QpOptions,
    ) -> OptimizeResult<QpResult<WgpuRuntime>> {
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
