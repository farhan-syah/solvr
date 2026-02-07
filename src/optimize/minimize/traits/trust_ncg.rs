//! Trust-NCG algorithm trait.

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::trust_region::{TrustRegionOptions, TrustRegionResult};
use crate::optimize::error::OptimizeResult;

/// Trust-NCG: Trust region with Steihaug-Toint conjugate gradient subproblem.
///
/// Uses CG to approximately solve the trust region subproblem. Handles
/// negative curvature by truncating at the trust region boundary.
/// Memory: O(n) per iteration.
pub trait TrustNcgAlgorithms<R: Runtime> {
    fn trust_ncg<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &TrustRegionOptions,
    ) -> OptimizeResult<TrustRegionResult<R>>
    where
        F: Fn(&Var<R>, &Self) -> NumrResult<Var<R>>;
}
