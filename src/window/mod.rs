//! Window function algorithms for signal processing
//!
//! This module defines the [`WindowFunctions`] trait that provides window functions
//! commonly used in spectral analysis and signal processing. All window functions
//! use the periodic formulation (suitable for FFT-based analysis).
//!
//! # Supported Windows
//!
//! - **Hann (Hanning)**: General-purpose window with good frequency resolution
//! - **Hamming**: Similar to Hann but with better sidelobe suppression
//! - **Blackman**: Excellent sidelobe suppression at the cost of main lobe width
//! - **Kaiser**: Parametric window with adjustable frequency resolution vs sidelobe tradeoff
//!
//! # Window Function Comparison
//!
//! | Window     | First Sidelobe | Sidelobe Rolloff | Main Lobe Width | Best For |
//! |------------|---------------|------------------|-----------------|----------|
//! | Rectangular| -13 dB        | -6 dB/octave     | Narrowest       | Transient analysis |
//! | Hann       | -31.5 dB      | -18 dB/octave    | Moderate        | General purpose |
//! | Hamming    | -42.7 dB      | -6 dB/octave     | Moderate        | Audio processing |
//! | Blackman   | -58 dB        | -18 dB/octave    | Wide            | High dynamic range |
//! | Kaiser     | Adjustable    | Adjustable       | Adjustable      | Custom requirements |
//!
//! # Choosing a Window
//!
//! - **Hann**: Start here. Best all-around choice for most applications.
//! - **Hamming**: Use when you need consistent sidelobe attenuation (-42 dB floor).
//! - **Blackman**: Use when spectral leakage must be minimized (e.g., detecting weak signals).
//! - **Kaiser**: Use when you need precise control over the resolution/leakage tradeoff.
//!
//! ## Kaiser Beta Guidelines
//!
//! The Kaiser window's `beta` parameter controls the tradeoff:
//!
//! | Beta | Approximate Sidelobe | Equivalent Window |
//! |------|---------------------|-------------------|
//! | 0    | -13 dB              | Rectangular       |
//! | 5    | -50 dB              | Hamming           |
//! | 6    | -60 dB              | Hann              |
//! | 8.6  | -90 dB              | Blackman          |
//! | 14   | -120 dB             | (very narrow)     |
//!
//! # Implementation Notes
//!
//! Window functions are implemented on CPU regardless of the target device, as they are
//! typically small arrays where GPU acceleration provides no benefit. The generated
//! window is transferred to the target device when needed.
//!
//! # Mathematical Definitions (Periodic Formulation)
//!
//! For a window of size N with n = 0, 1, ..., N-1:
//!
//! ```text
//! Hann:     w[n] = 0.5 - 0.5 * cos(2*pi*n / N)
//! Hamming:  w[n] = 0.54 - 0.46 * cos(2*pi*n / N)
//! Blackman: w[n] = 0.42 - 0.5 * cos(2*pi*n / N) + 0.08 * cos(4*pi*n / N)
//! Kaiser:   w[n] = I0(beta * sqrt(1 - ((n - N/2) / (N/2))^2)) / I0(beta)
//! ```
//!
//! Where I0 is the modified Bessel function of the first kind, order 0.

mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
pub mod generators;
#[cfg(feature = "wgpu")]
mod wgpu;

use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

// Re-export commonly used generator functions
pub use generators::{
    bessel_i0, generate_blackman_f64, generate_hamming_f64, generate_hann_f64, generate_kaiser_f64,
};

/// Trait for generating window functions used in signal processing.
///
/// Window functions are multiplied with a signal before FFT analysis to reduce
/// spectral leakage. Different windows offer different tradeoffs between
/// frequency resolution and sidelobe suppression.
///
/// # Backend Implementation
///
/// All backends use CPU-based generation since window functions are typically
/// small arrays. The result is transferred to the target device if necessary.
///
/// # Example
///
/// ```ignore
/// use solvr::window::WindowFunctions;
/// use numr::runtime::cpu::{CpuClient, CpuDevice};
/// use numr::dtype::DType;
///
/// let device = CpuDevice::new();
/// let client = CpuClient::new(device.clone());
/// let window = client.hann_window(1024, DType::F32, &device)?;
/// ```
pub trait WindowFunctions<R: Runtime> {
    /// Generate a Hann (Hanning) window.
    ///
    /// The Hann window is a general-purpose window with good frequency resolution
    /// and moderate sidelobe suppression (-31.5 dB first sidelobe).
    ///
    /// # Formula (periodic)
    ///
    /// ```text
    /// w[n] = 0.5 - 0.5 * cos(2*pi*n / N)
    /// ```
    fn hann_window(&self, size: usize, dtype: DType, device: &R::Device) -> Result<Tensor<R>>;

    /// Generate a Hamming window.
    ///
    /// The Hamming window has better sidelobe suppression than Hann (-42.7 dB first
    /// sidelobe) but doesn't go to zero at the edges.
    ///
    /// # Formula (periodic)
    ///
    /// ```text
    /// w[n] = 0.54 - 0.46 * cos(2*pi*n / N)
    /// ```
    fn hamming_window(&self, size: usize, dtype: DType, device: &R::Device) -> Result<Tensor<R>>;

    /// Generate a Blackman window.
    ///
    /// The Blackman window has excellent sidelobe suppression (-58 dB first sidelobe)
    /// at the cost of wider main lobe than Hann/Hamming.
    ///
    /// # Formula (periodic)
    ///
    /// ```text
    /// w[n] = 0.42 - 0.5 * cos(2*pi*n / N) + 0.08 * cos(4*pi*n / N)
    /// ```
    fn blackman_window(&self, size: usize, dtype: DType, device: &R::Device) -> Result<Tensor<R>>;

    /// Generate a Kaiser window.
    ///
    /// The Kaiser window is a flexible window with adjustable parameter beta that
    /// controls the tradeoff between main lobe width and sidelobe level.
    ///
    /// # Formula
    ///
    /// ```text
    /// w[n] = I0(beta * sqrt(1 - ((n - N/2) / (N/2))^2)) / I0(beta)
    /// ```
    ///
    /// # Beta Parameter Guidelines
    ///
    /// | Beta  | Sidelobe Attenuation | Approximate Equivalent |
    /// |-------|---------------------|------------------------|
    /// | 0     | -13 dB             | Rectangular            |
    /// | 5     | -50 dB             | Hamming                |
    /// | 6     | -60 dB             | Hann                   |
    /// | 8.6   | -90 dB             | Blackman               |
    fn kaiser_window(
        &self,
        size: usize,
        beta: f64,
        dtype: DType,
        device: &R::Device,
    ) -> Result<Tensor<R>>;
}

/// Validate window dtype (must be F32 or F64).
pub fn validate_window_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Validate window size (must be positive).
pub fn validate_window_size(size: usize, op: &'static str) -> Result<()> {
    if size == 0 {
        return Err(Error::InvalidArgument {
            arg: "size",
            reason: format!("{op} requires size > 0"),
        });
    }
    Ok(())
}
