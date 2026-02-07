//! Generic signal processing implementations.
//!
//! This module provides Runtime-generic implementations of signal processing
//! algorithms. All functions work with any numr backend (CPU, CUDA, WebGPU).
//!
//! # Architecture
//!
//! All signal processing operations are fully tensor-based - data stays on device
//! with no GPU->CPU->GPU roundtrips in algorithm loops. Operations use numr's
//! tensor ops: `narrow`, `pad`, `mul`, `add`, `cat`, `rfft`, `irfft`, etc.
//!
//! The key benefit: **zero code duplication** across backends. CPU, CUDA, and
//! WebGPU all use these same implementations.

mod analysis;
mod boundary;
mod convolution;
mod edge;
mod frequency_response;
mod helpers;
mod kernels;
mod nd_filters;
mod padding;
mod slice;
mod spectral;
mod stft;

// Re-export only what backends need
// Boundary mode padding for N-D filters:
pub use boundary::pad_axis_impl;
// Filter kernel generation:
pub use kernels::{edge_kernel_1d, gaussian_kernel_1d, laplace_kernel_1d, uniform_kernel_1d};
// GPU-accelerable algorithms:
pub use analysis::{hilbert_impl, resample_impl};
// CPU helper functions used by cpu/ implementations:
pub use analysis::{
    apply_butter_lowpass, apply_fir_lowpass, compute_prominences, compute_savgol_coeffs,
    filter_by_distance,
};
// N-D filter algorithms:
pub use nd_filters::{
    gaussian_filter_impl, maximum_filter_impl, minimum_filter_impl, percentile_filter_impl,
    uniform_filter_impl,
};
// CPU-only algorithms (decimate, find_peaks, savgol, extrema, medfilt, wiener) live in cpu/
pub use convolution::{convolve_impl, convolve2d_impl, correlate_impl, correlate2d_impl};
pub use edge::{
    gaussian_gradient_magnitude_impl, gaussian_laplace_impl, laplace_impl, prewitt_impl, sobel_impl,
};
pub use frequency_response::{freqz_impl, group_delay_impl, sosfreqz_impl};
pub use spectral::{coherence_impl, csd_impl, lombscargle_impl, periodogram_impl, welch_impl};
pub use stft::{istft_impl, spectrogram_impl, stft_impl};

// Note: filter_apply (lfilter, filtfilt, sosfilt, sosfiltfilt) is CPU-only
// because IIR filtering is inherently sequential. See cpu/filter_apply.rs.
