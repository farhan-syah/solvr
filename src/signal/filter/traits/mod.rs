//! Filter algorithm traits.

pub mod conversions;
pub mod fir_design;
pub mod iir_design;

pub use conversions::FilterConversions;
pub use fir_design::FirDesignAlgorithms;
pub use iir_design::IirDesignAlgorithms;
