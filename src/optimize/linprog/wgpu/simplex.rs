//! WebGPU implementation of simplex linear programming.

use crate::optimize::linprog::impl_generic::TensorLinearConstraints;
use crate::optimize::linprog::impl_generic::simplex_impl;
use crate::optimize::linprog::{
    LinProgAlgorithms, LinProgOptions, LinProgTensorConstraints, LinProgTensorResult,
};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl LinProgAlgorithms<WgpuRuntime> for WgpuClient {
    fn linprog(
        &self,
        c: &Tensor<WgpuRuntime>,
        constraints: &LinProgTensorConstraints<WgpuRuntime>,
        options: &LinProgOptions,
    ) -> Result<LinProgTensorResult<WgpuRuntime>> {
        let tensor_constraints = TensorLinearConstraints {
            a_ub: constraints.a_ub.clone(),
            b_ub: constraints.b_ub.clone(),
            a_eq: constraints.a_eq.clone(),
            b_eq: constraints.b_eq.clone(),
            lower_bounds: constraints.lower_bounds.clone(),
            upper_bounds: constraints.upper_bounds.clone(),
        };

        let result = simplex_impl(self, c, &tensor_constraints, options).map_err(|e| {
            numr::error::Error::backend_limitation("wgpu", "linprog", e.to_string())
        })?;

        Ok(LinProgTensorResult {
            x: result.x,
            fun: result.fun,
            success: result.success,
            nit: result.nit,
            message: result.message,
            slack: result.slack,
        })
    }
}
