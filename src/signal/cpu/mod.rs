//! CPU implementation of signal processing algorithms.
//!
//! This module implements the [`SignalProcessingAlgorithms`] trait for CPU
//! by delegating to the generic implementations in `impl_generic/`.

use crate::signal::impl_generic::{
    convolve_impl, convolve2d_impl, correlate_impl, correlate2d_impl, istft_impl, spectrogram_impl,
    stft_impl,
};
use crate::signal::{ConvMode, SignalProcessingAlgorithms};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl SignalProcessingAlgorithms<CpuRuntime> for CpuClient {
    fn convolve(
        &self,
        signal: &Tensor<CpuRuntime>,
        kernel: &Tensor<CpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CpuRuntime>> {
        convolve_impl(self, signal, kernel, mode)
    }

    fn convolve2d(
        &self,
        signal: &Tensor<CpuRuntime>,
        kernel: &Tensor<CpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CpuRuntime>> {
        convolve2d_impl(self, signal, kernel, mode)
    }

    fn correlate(
        &self,
        signal: &Tensor<CpuRuntime>,
        kernel: &Tensor<CpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CpuRuntime>> {
        correlate_impl(self, signal, kernel, mode)
    }

    fn correlate2d(
        &self,
        signal: &Tensor<CpuRuntime>,
        kernel: &Tensor<CpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CpuRuntime>> {
        correlate2d_impl(self, signal, kernel, mode)
    }

    fn stft(
        &self,
        signal: &Tensor<CpuRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<CpuRuntime>>,
        center: bool,
        normalized: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        stft_impl(self, signal, n_fft, hop_length, window, center, normalized)
    }

    fn istft(
        &self,
        stft_matrix: &Tensor<CpuRuntime>,
        hop_length: Option<usize>,
        window: Option<&Tensor<CpuRuntime>>,
        center: bool,
        length: Option<usize>,
        normalized: bool,
    ) -> Result<Tensor<CpuRuntime>> {
        istft_impl(
            self,
            stft_matrix,
            hop_length,
            window,
            center,
            length,
            normalized,
        )
    }

    fn spectrogram(
        &self,
        signal: &Tensor<CpuRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<CpuRuntime>>,
        power: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        spectrogram_impl(self, signal, n_fft, hop_length, window, power)
    }
}

#[cfg(test)]
mod tests;
