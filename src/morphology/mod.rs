//! Morphological operations for N-dimensional arrays.
//!
//! Provides binary and grey-scale morphological operations (erosion, dilation,
//! opening, closing) and connected component labeling with region measurements.

mod cpu;
pub mod impl_generic;
pub mod traits;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "wgpu")]
mod wgpu;

pub use traits::binary::{BinaryMorphologyAlgorithms, StructuringElement};
pub use traits::grey::GreyMorphologyAlgorithms;
pub use traits::measurements::{MeasurementAlgorithms, RegionProperties};
