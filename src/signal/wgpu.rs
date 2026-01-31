//! WebGPU implementation of signal processing algorithms.
//!
//! This module implements the [`SignalProcessingAlgorithms`] trait for WebGPU
//! by delegating to the generic implementations in `impl_generic/`.
//!
//! # Limitations
//!
//! - Only F32 is supported (WGSL doesn't support F64)

use crate::signal::impl_generic::{
    convolve_impl, convolve2d_impl, correlate_impl, correlate2d_impl, istft_impl, spectrogram_impl,
    stft_impl,
};
use crate::signal::{ConvMode, SignalProcessingAlgorithms};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl SignalProcessingAlgorithms<WgpuRuntime> for WgpuClient {
    fn convolve(
        &self,
        signal: &Tensor<WgpuRuntime>,
        kernel: &Tensor<WgpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<WgpuRuntime>> {
        convolve_impl(self, signal, kernel, mode)
    }

    fn convolve2d(
        &self,
        signal: &Tensor<WgpuRuntime>,
        kernel: &Tensor<WgpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<WgpuRuntime>> {
        convolve2d_impl(self, signal, kernel, mode)
    }

    fn correlate(
        &self,
        signal: &Tensor<WgpuRuntime>,
        kernel: &Tensor<WgpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<WgpuRuntime>> {
        correlate_impl(self, signal, kernel, mode)
    }

    fn correlate2d(
        &self,
        signal: &Tensor<WgpuRuntime>,
        kernel: &Tensor<WgpuRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<WgpuRuntime>> {
        correlate2d_impl(self, signal, kernel, mode)
    }

    fn stft(
        &self,
        signal: &Tensor<WgpuRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<WgpuRuntime>>,
        center: bool,
        normalized: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        stft_impl(self, signal, n_fft, hop_length, window, center, normalized)
    }

    fn istft(
        &self,
        stft_matrix: &Tensor<WgpuRuntime>,
        hop_length: Option<usize>,
        window: Option<&Tensor<WgpuRuntime>>,
        center: bool,
        length: Option<usize>,
        normalized: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
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
        signal: &Tensor<WgpuRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<WgpuRuntime>>,
        power: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        spectrogram_impl(self, signal, n_fft, hop_length, window, power)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::wgpu::WgpuDevice;

    fn setup() -> Option<(WgpuClient, WgpuDevice)> {
        // Skip if no WebGPU device available
        let device = WgpuDevice::new().ok()?;
        let client = WgpuClient::new(device.clone());
        Some((client, device))
    }

    #[test]
    fn test_convolve_wgpu() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping WebGPU test: no device available");
            return;
        };

        // WebGPU only supports F32
        let signal =
            Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let kernel = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device);

        let result = client.convolve(&signal, &kernel, ConvMode::Full).unwrap();

        assert_eq!(result.shape(), &[7]);
    }

    #[test]
    fn test_stft_wgpu() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping WebGPU test: no device available");
            return;
        };

        let signal: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let signal_tensor = Tensor::<WgpuRuntime>::from_slice(&signal, &[256], &device);

        let result = client
            .stft(&signal_tensor, 64, Some(16), None, true, false)
            .unwrap();

        let freq_bins = 64 / 2 + 1;
        let n_frames = (256 + 64 - 64) / 16 + 1;

        assert_eq!(result.shape(), &[n_frames, freq_bins]);
    }

    #[test]
    fn test_spectrogram_wgpu() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping WebGPU test: no device available");
            return;
        };

        let signal: Vec<f32> = (0..512).map(|i| (i as f32 * 0.05).sin()).collect();
        let signal_tensor = Tensor::<WgpuRuntime>::from_slice(&signal, &[512], &device);

        let result = client
            .spectrogram(&signal_tensor, 64, Some(32), None, 2.0)
            .unwrap();

        let freq_bins = 64 / 2 + 1;
        let n_frames = (512 + 64 - 64) / 32 + 1;
        assert_eq!(result.shape(), &[n_frames, freq_bins]);
    }
}
