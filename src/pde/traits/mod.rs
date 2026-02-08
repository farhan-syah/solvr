//! PDE algorithm trait definitions.

pub mod finite_difference;
pub mod finite_element;
pub mod spectral;

pub use finite_difference::FiniteDifferenceAlgorithms;
pub use finite_element::FiniteElementAlgorithms;
pub use spectral::SpectralAlgorithms;
