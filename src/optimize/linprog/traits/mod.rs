//! Trait definitions for linear programming algorithms.

mod milp;
mod simplex;

pub use milp::{MilpAlgorithms, MilpOptions, MilpTensorResult};
pub use simplex::{
    LinProgAlgorithms, LinProgOptions, LinProgTensorConstraints, LinProgTensorResult,
};
