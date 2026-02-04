//! Generic implementations of spectral analysis algorithms.

pub mod coherence;
pub mod csd;
pub mod helpers;
pub mod lombscargle;
pub mod periodogram;
pub mod welch;

pub use coherence::coherence_impl;
pub use csd::csd_impl;
pub use lombscargle::lombscargle_impl;
pub use periodogram::periodogram_impl;
pub use welch::welch_impl;
