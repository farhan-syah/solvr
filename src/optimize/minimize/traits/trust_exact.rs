//! Trust-Exact algorithm trait.
use crate::DType;

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::trust_region::{TrustRegionOptions, TrustRegionResult};
use crate::optimize::error::OptimizeResult;

/// Trust-Exact: Trust region with nearly-exact subproblem solution.
///
/// Solves the subproblem by finding lambda such that ||(H+lambda*I)^{-1}g|| = delta.
/// Uses Cholesky factorizations. Handles the "hard case".
/// Memory: O(n^2) for full Hessian.
pub trait TrustExactAlgorithms<R: Runtime<DType = DType>> {
    fn trust_exact<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &TrustRegionOptions,
    ) -> OptimizeResult<TrustRegionResult<R>>
    where
        F: Fn(&Var<R>, &Self) -> NumrResult<Var<R>>;
}
