//! Trait definitions for linear programming algorithms.

mod simplex;
mod milp;

pub use simplex::{LinProgAlgorithms, LinProgOptions, LinProgTensorConstraints, LinProgTensorResult};
pub use milp::{MilpAlgorithms, MilpOptions, MilpTensorResult};
