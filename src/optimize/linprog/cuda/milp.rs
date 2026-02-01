//! CUDA implementation of mixed-integer linear programming.

use crate::optimize::linprog::impl_generic::{milp_impl, MilpOptionsInternal, TensorLinearConstraints};
use crate::optimize::linprog::{
    LinProgOptions, LinProgTensorConstraints, MilpAlgorithms, MilpOptions, MilpTensorResult,
};
use numr::error::Result;
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl MilpAlgorithms<CudaRuntime> for CudaClient {
    fn milp(
        &self,
        c: &Tensor<CudaRuntime>,
        constraints: &LinProgTensorConstraints<CudaRuntime>,
        integrality: &Tensor<CudaRuntime>,
        options: &MilpOptions,
    ) -> Result<MilpTensorResult<CudaRuntime>> {
        // Convert constraints to impl_generic format
        let internal_constraints = TensorLinearConstraints {
            a_ub: constraints.a_ub.clone(),
            b_ub: constraints.b_ub.clone(),
            a_eq: constraints.a_eq.clone(),
            b_eq: constraints.b_eq.clone(),
            lower_bounds: constraints.lower_bounds.clone(),
            upper_bounds: constraints.upper_bounds.clone(),
        };

        let internal_options = MilpOptionsInternal {
            max_nodes: options.max_nodes,
            int_tol: options.int_tol,
            gap_tol: options.gap_tol,
            lp_options: options.lp_options.clone(),
        };

        let result = milp_impl(self, c, &internal_constraints, integrality, &internal_options)
            .map_err(|e| numr::error::Error::backend_limitation("cuda", "milp", e.to_string()))?;

        Ok(MilpTensorResult {
            x: result.x,
            fun: result.fun,
            success: result.success,
            nodes: result.nodes,
            gap: result.gap,
            message: result.message,
        })
    }
}
