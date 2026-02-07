//! Morphological algorithm traits.

pub mod binary;
pub mod grey;
pub mod measurements;

pub use binary::{BinaryMorphologyAlgorithms, StructuringElement};
pub use grey::GreyMorphologyAlgorithms;
pub use measurements::{MeasurementAlgorithms, RegionProperties};
