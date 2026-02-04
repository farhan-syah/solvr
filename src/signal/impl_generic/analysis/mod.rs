//! Generic implementations of signal analysis algorithms.

pub mod decimate;
pub mod find_peaks;
pub mod helpers;
pub mod hilbert;
pub mod resample;
pub mod savgol;

pub use decimate::decimate_impl;
pub use find_peaks::find_peaks_impl;
pub use hilbert::hilbert_impl;
pub use resample::resample_impl;
pub use savgol::savgol_filter_impl;
