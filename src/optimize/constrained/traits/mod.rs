//! Trait definitions for constrained optimization algorithms.

pub mod cobyla;
pub mod slsqp;
pub mod trust_constr;
pub mod types;

pub use cobyla::CobylaAlgorithms;
pub use slsqp::SlsqpAlgorithms;
pub use trust_constr::TrustConstrAlgorithms;
pub use types::{Bounds, ConstrainedOptions, ConstrainedResult, Constraint, ConstraintType};
