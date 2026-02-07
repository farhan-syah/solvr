//! Trait definitions and types for multivariate minimization.

pub mod newton_cg;
pub mod trust_exact;
pub mod trust_krylov;
pub mod trust_ncg;
pub mod trust_region;
mod types;

pub use newton_cg::{NewtonCGAlgorithms, NewtonCGOptions, NewtonCGResult};
pub use trust_exact::TrustExactAlgorithms;
pub use trust_krylov::TrustKrylovAlgorithms;
pub use trust_ncg::TrustNcgAlgorithms;
pub use trust_region::{TrustRegionOptions, TrustRegionResult};
pub use types::{MinimizeOptions, TensorMinimizeResult};
