//! Generic signal processing implementations.
//!
//! This module provides Runtime-generic implementations of signal processing
//! algorithms. All functions work with any numr backend (CPU, CUDA, WebGPU).
//!
//! # Architecture
//!
//! Unlike the stats module which can use TensorOps directly for all computations,
//! signal processing requires several operations not available as TensorOps:
//! - Complex element-wise multiplication
//! - Tensor reversal/flip
//! - Arbitrary padding
//! - Arbitrary slicing
//!
//! These are implemented using `tensor.to_vec()` and `Tensor::from_slice()` as
//! a universal fallback. This means data transfers occur for these operations,
//! but the FFT computations (the expensive part) stay on device.
//!
//! The key benefit: **zero code duplication** across backends. CPU, CUDA, and
//! WebGPU all use these same implementations.

mod convolution;
mod helpers;
mod padding;
mod slice;
mod stft;

// Re-export only what backends need - internal helpers are used within impl_generic
pub use convolution::{convolve_impl, convolve2d_impl, correlate_impl, correlate2d_impl};
pub use stft::{istft_impl, spectrogram_impl, stft_impl};
