//! WebGPU implementations of constrained optimization algorithms.

pub mod cobyla;
pub mod slsqp;
pub mod trust_constr;

pub use cobyla::*;
pub use slsqp::*;
pub use trust_constr::*;
