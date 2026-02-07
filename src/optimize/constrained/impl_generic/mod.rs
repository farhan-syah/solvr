//! Generic implementations of constrained optimization algorithms.

pub mod cobyla;
pub mod qp_subproblem;
pub mod slsqp;
pub mod trust_constr;

pub use cobyla::cobyla_impl;
pub use qp_subproblem::qp_subproblem_impl;
pub use slsqp::slsqp_impl;
pub use trust_constr::trust_constr_impl;
