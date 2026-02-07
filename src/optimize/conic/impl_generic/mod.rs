//! Generic implementations of conic programming algorithms.

pub mod sdp;
pub mod socp;

pub use sdp::sdp_impl;
pub use socp::socp_impl;
