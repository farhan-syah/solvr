//! CUDA implementation of signal processing algorithms.
//!
//! This module implements the [`SignalProcessingAlgorithms`] trait for CUDA
//! by delegating to the generic implementations in `impl_generic/`.

use crate::signal::impl_generic::{
    convolve_impl, convolve2d_impl, correlate_impl, correlate2d_impl, istft_impl, spectrogram_impl,
    stft_impl,
};
use crate::signal::{ConvMode, SignalProcessingAlgorithms};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl SignalProcessingAlgorithms<CudaRuntime> for CudaClient {
    fn convolve(
        &self,
        signal: &Tensor<CudaRuntime>,
        kernel: &Tensor<CudaRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CudaRuntime>> {
        convolve_impl(self, signal, kernel, mode)
    }

    fn convolve2d(
        &self,
        signal: &Tensor<CudaRuntime>,
        kernel: &Tensor<CudaRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CudaRuntime>> {
        convolve2d_impl(self, signal, kernel, mode)
    }

    fn correlate(
        &self,
        signal: &Tensor<CudaRuntime>,
        kernel: &Tensor<CudaRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CudaRuntime>> {
        correlate_impl(self, signal, kernel, mode)
    }

    fn correlate2d(
        &self,
        signal: &Tensor<CudaRuntime>,
        kernel: &Tensor<CudaRuntime>,
        mode: ConvMode,
    ) -> Result<Tensor<CudaRuntime>> {
        correlate2d_impl(self, signal, kernel, mode)
    }

    fn stft(
        &self,
        signal: &Tensor<CudaRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<CudaRuntime>>,
        center: bool,
        normalized: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        stft_impl(self, signal, n_fft, hop_length, window, center, normalized)
    }

    fn istft(
        &self,
        stft_matrix: &Tensor<CudaRuntime>,
        hop_length: Option<usize>,
        window: Option<&Tensor<CudaRuntime>>,
        center: bool,
        length: Option<usize>,
        normalized: bool,
    ) -> Result<Tensor<CudaRuntime>> {
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
        signal: &Tensor<CudaRuntime>,
        n_fft: usize,
        hop_length: Option<usize>,
        window: Option<&Tensor<CudaRuntime>>,
        power: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        spectrogram_impl(self, signal, n_fft, hop_length, window, power)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaClient, CudaDevice)> {
        // Skip if no CUDA device available
        let device = CudaDevice::new(0).ok()?;
        let client = CudaClient::new(device.clone());
        Some((client, device))
    }

    #[test]
    fn test_convolve_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let signal =
            Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let kernel = Tensor::<CudaRuntime>::from_slice(&[1.0f64, 1.0, 1.0], &[3], &device);

        let result = client.convolve(&signal, &kernel, ConvMode::Full).unwrap();

        assert_eq!(result.shape(), &[7]);
    }

    #[test]
    fn test_stft_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let signal: Vec<f64> = (0..256).map(|i| (i as f64 * 0.1).sin()).collect();
        let signal_tensor = Tensor::<CudaRuntime>::from_slice(&signal, &[256], &device);

        let result = client
            .stft(&signal_tensor, 64, Some(16), None, true, false)
            .unwrap();

        let freq_bins = 64 / 2 + 1;
        let n_frames = (256 + 64 - 64) / 16 + 1;

        assert_eq!(result.shape(), &[n_frames, freq_bins]);
    }

    #[test]
    fn test_spectrogram_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let signal: Vec<f64> = (0..512).map(|i| (i as f64 * 0.05).sin()).collect();
        let signal_tensor = Tensor::<CudaRuntime>::from_slice(&signal, &[512], &device);

        let result = client
            .spectrogram(&signal_tensor, 64, Some(32), None, 2.0)
            .unwrap();

        let freq_bins = 64 / 2 + 1;
        let n_frames = (512 + 64 - 64) / 32 + 1;
        assert_eq!(result.shape(), &[n_frames, freq_bins]);
    }
}
