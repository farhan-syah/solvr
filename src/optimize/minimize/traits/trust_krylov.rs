//! Trust-Krylov algorithm trait.

use numr::autograd::Var;
use numr::error::Result as NumrResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::trust_region::{TrustRegionOptions, TrustRegionResult};
use crate::optimize::error::OptimizeResult;

/// Trust-Krylov: Trust region with GLTR (Generalized Lanczos Trust Region).
///
/// Uses Lanczos iteration to build a tridiagonal approximation of H,
/// then solves the trust region subproblem in the reduced Krylov space.
/// Memory: O(kn) where k is the Lanczos iteration count.
pub trait TrustKrylovAlgorithms<R: Runtime> {
    fn trust_krylov<F>(
        &self,
        f: F,
        x0: &Tensor<R>,
        options: &TrustRegionOptions,
    ) -> OptimizeResult<TrustRegionResult<R>>
    where
        F: Fn(&Var<R>, &Self) -> NumrResult<Var<R>>;
}
