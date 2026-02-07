//! Generic implementations of statistical algorithms.
//!
//! These implementations work with any numr Runtime backend.
//! Each backend-specific module (cpu, cuda, wgpu) delegates to these.

mod descriptive;
mod hypothesis;
mod information;
mod regression;
mod robust;

pub use descriptive::*;
pub use hypothesis::*;
pub use information::*;
pub use regression::*;
pub use robust::*;
