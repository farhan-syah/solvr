//! Mixed-integer linear programming trait and types.

use super::simplex::{LinProgOptions, LinProgTensorConstraints};
use numr::error::Result;
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Options for mixed-integer linear programming.
#[derive(Debug, Clone)]
pub struct MilpOptions {
    /// Maximum number of nodes to explore in branch-and-bound.
    pub max_nodes: usize,
    /// Tolerance for integer feasibility.
    pub int_tol: f64,
    /// Tolerance for optimality gap.
    pub gap_tol: f64,
    /// Base LP solver options.
    pub lp_options: LinProgOptions,
}

impl Default for MilpOptions {
    fn default() -> Self {
        Self {
            max_nodes: 10000,
            int_tol: 1e-6,
            gap_tol: 1e-4,
            lp_options: LinProgOptions::default(),
        }
    }
}

/// Result from tensor-based mixed-integer linear programming.
#[derive(Debug, Clone)]
pub struct MilpTensorResult<R: Runtime> {
    /// Optimal solution vector.
    pub x: Tensor<R>,
    /// Optimal objective value.
    pub fun: f64,
    /// Whether optimization succeeded.
    pub success: bool,
    /// Number of nodes explored.
    pub nodes: usize,
    /// Optimality gap (upper_bound - lower_bound) / |upper_bound|.
    pub gap: f64,
    /// Status message.
    pub message: String,
}

/// Algorithmic contract for mixed-integer linear programming.
pub trait MilpAlgorithms<R: Runtime>: TensorOps<R> + ScalarOps<R> + CompareOps<R> + RuntimeClient<R> {
    /// Solve a mixed-integer linear programming problem using branch-and-bound.
    ///
    /// # Arguments
    ///
    /// * `c` - Objective function coefficients
    /// * `constraints` - Linear constraints
    /// * `integrality` - Tensor of booleans (as f64: 1.0 = integer, 0.0 = continuous)
    /// * `options` - Solver options
    fn milp(
        &self,
        c: &Tensor<R>,
        constraints: &LinProgTensorConstraints<R>,
        integrality: &Tensor<R>,
        options: &MilpOptions,
    ) -> Result<MilpTensorResult<R>>;
}
