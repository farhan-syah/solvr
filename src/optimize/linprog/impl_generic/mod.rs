//! Generic implementations for linear programming algorithms.
use crate::DType;

mod milp;
mod simplex;

pub use milp::{MilpOptionsInternal, milp_impl};
pub use simplex::{TensorLinProgResult, simplex_impl};

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Tensor-based linear constraints.
#[derive(Debug, Clone)]
pub struct TensorLinearConstraints<R: Runtime<DType = DType>> {
    /// Inequality constraint matrix (A_ub * x <= b_ub)
    pub a_ub: Option<Tensor<R>>,
    /// Inequality constraint bounds
    pub b_ub: Option<Tensor<R>>,
    /// Equality constraint matrix (A_eq * x == b_eq)
    pub a_eq: Option<Tensor<R>>,
    /// Equality constraint bounds
    pub b_eq: Option<Tensor<R>>,
    /// Variable lower bounds
    pub lower_bounds: Option<Tensor<R>>,
    /// Variable upper bounds
    pub upper_bounds: Option<Tensor<R>>,
}
