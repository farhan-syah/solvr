//! Generic implementations of QP algorithms.

pub mod active_set;
pub mod interior_point;

pub use active_set::active_set_qp_impl;
pub use interior_point::interior_point_qp_impl;
