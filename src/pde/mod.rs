mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
pub mod error;
pub mod impl_generic;
pub mod traits;
pub mod types;
#[cfg(feature = "wgpu")]
mod wgpu;

pub use error::{PdeError, PdeResult};
pub use traits::{FiniteDifferenceAlgorithms, FiniteElementAlgorithms, SpectralAlgorithms};
pub use types::{
    BoundaryCondition, BoundarySide, BoundarySpec, FdmOptions, FdmResult, FemResult, Grid2D,
    Grid3D, Preconditioner, SparseSolver, SpectralResult, TimeDependentOptions, TimeResult,
};
