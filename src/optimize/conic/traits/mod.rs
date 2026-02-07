//! Trait definitions for conic programming.

pub mod sdp;
pub mod socp;

pub use sdp::{SdpAlgorithms, SdpOptions, SdpResult};
pub use socp::{SocConstraint, SocpAlgorithms, SocpOptions, SocpResult};
