//! Generic implementations of global optimization algorithms.

mod differential_evolution;
mod simulated_annealing;

pub use differential_evolution::differential_evolution_impl;
pub use simulated_annealing::simulated_annealing_impl;

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Result from tensor-based global optimization.
#[derive(Debug, Clone)]
pub struct TensorGlobalResult<R: Runtime> {
    pub x: Tensor<R>,
    pub fun: f64,
    pub iterations: usize,
    pub nfev: usize,
    pub converged: bool,
}
