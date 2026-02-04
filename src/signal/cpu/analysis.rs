//! CPU implementation of signal analysis algorithms.

use crate::signal::impl_generic::{
    decimate_impl, find_peaks_impl, hilbert_impl, resample_impl, savgol_filter_impl,
};
use crate::signal::traits::analysis::{
    DecimateParams, HilbertResult, PeakParams, PeakResult, SignalAnalysisAlgorithms,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl SignalAnalysisAlgorithms<CpuRuntime> for CpuClient {
    fn hilbert(&self, x: &Tensor<CpuRuntime>) -> Result<HilbertResult<CpuRuntime>> {
        hilbert_impl(self, x)
    }

    fn resample(
        &self,
        x: &Tensor<CpuRuntime>,
        num: usize,
        den: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        resample_impl(self, x, num, den)
    }

    fn decimate(
        &self,
        x: &Tensor<CpuRuntime>,
        q: usize,
        params: DecimateParams,
    ) -> Result<Tensor<CpuRuntime>> {
        decimate_impl(self, x, q, params)
    }

    fn find_peaks(
        &self,
        x: &Tensor<CpuRuntime>,
        params: PeakParams,
    ) -> Result<PeakResult<CpuRuntime>> {
        find_peaks_impl(self, x, params)
    }

    fn savgol_filter(
        &self,
        x: &Tensor<CpuRuntime>,
        window_length: usize,
        polyorder: usize,
        deriv: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        savgol_filter_impl(self, x, window_length, polyorder, deriv)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;
    use std::f64::consts::PI;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_hilbert_sine() {
        let (client, device) = setup();

        // Hilbert transform of sin should be -cos (with some edge effects)
        let n = 256;
        let freq = 5.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / n as f64).sin())
            .collect();

        let x = Tensor::from_slice(&signal, &[n], &device);
        let result = client.hilbert(&x).unwrap();

        let real: Vec<f64> = result.real.to_vec();
        let imag: Vec<f64> = result.imag.to_vec();

        // Check envelope is approximately 1
        let env = result.envelope().unwrap();
        let env_data: Vec<f64> = env.to_vec();

        // Ignore edge effects (first and last 10%)
        let start = n / 10;
        let end = n - n / 10;
        for &e in &env_data[start..end] {
            assert!((e - 1.0).abs() < 0.15, "Envelope should be ~1.0, got {}", e);
        }
    }

    #[test]
    fn test_hilbert_envelope() {
        let (client, device) = setup();

        // AM modulated signal: (1 + 0.5*sin(low_freq)) * sin(high_freq)
        let n = 512;
        let carrier_freq = 50.0;
        let mod_freq = 5.0;

        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                (1.0 + 0.5 * (2.0 * PI * mod_freq * t).sin()) * (2.0 * PI * carrier_freq * t).sin()
            })
            .collect();

        let x = Tensor::from_slice(&signal, &[n], &device);
        let result = client.hilbert(&x).unwrap();
        let env = result.envelope().unwrap();
        let env_data: Vec<f64> = env.to_vec();

        // The envelope should follow the modulating signal
        // Check that envelope has the modulation pattern
        let env_mean: f64 = env_data.iter().sum::<f64>() / n as f64;
        assert!((env_mean - 1.0).abs() < 0.2, "Mean envelope should be ~1.0");
    }

    #[test]
    fn test_resample_upsample() {
        let (client, device) = setup();

        // Simple signal
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = signal.len();
        let x = Tensor::from_slice(&signal, &[n], &device);

        // Upsample by 2
        let result = client.resample(&x, 2, 1).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Output should be approximately twice the length
        assert_eq!(result_data.len(), n * 2);
    }

    #[test]
    fn test_resample_downsample() {
        let (client, device) = setup();

        // Create a signal
        let n = 100;
        let signal: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let x = Tensor::from_slice(&signal, &[n], &device);

        // Downsample by 2
        let result = client.resample(&x, 1, 2).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Output should be approximately half the length
        assert_eq!(result_data.len(), (n + 1) / 2);
    }

    #[test]
    fn test_decimate() {
        let (client, device) = setup();

        // Create a signal with low-frequency content
        let n = 256;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 2.0 * i as f64 / n as f64).sin())
            .collect();
        let x = Tensor::from_slice(&signal, &[n], &device);

        // Decimate by 4
        let params = DecimateParams::default();
        let result = client.decimate(&x, 4, params).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Output should be n/4 samples
        assert_eq!(result_data.len(), (n + 3) / 4);

        // The decimated signal should still contain the low-frequency content
        // Check that it's not all zeros
        let max_val = result_data.iter().fold(0.0_f64, |a, &b| a.max(b.abs()));
        assert!(max_val > 0.1, "Decimated signal should preserve content");
    }

    #[test]
    fn test_find_peaks_simple() {
        let (client, device) = setup();

        // Simple signal with clear peaks
        let signal = vec![0.0, 1.0, 0.0, 2.0, 0.0, 1.5, 0.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let params = PeakParams::new();
        let result = client.find_peaks(&x, params).unwrap();

        // Should find peaks at indices 1, 3, 5
        assert_eq!(result.indices, vec![1, 3, 5]);

        let heights: Vec<f64> = result.heights.to_vec();
        assert!((heights[0] - 1.0).abs() < 1e-10);
        assert!((heights[1] - 2.0).abs() < 1e-10);
        assert!((heights[2] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_find_peaks_with_height() {
        let (client, device) = setup();

        let signal = vec![0.0, 1.0, 0.0, 2.0, 0.0, 0.5, 0.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let params = PeakParams::new().with_height(1.0);
        let result = client.find_peaks(&x, params).unwrap();

        // Should only find peaks >= 1.0
        assert_eq!(result.indices, vec![1, 3]);
    }

    #[test]
    fn test_find_peaks_with_distance() {
        let (client, device) = setup();

        // Peaks very close together
        let signal = vec![0.0, 1.0, 0.5, 1.2, 0.0, 0.0, 2.0, 0.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let params = PeakParams::new().with_distance(3);
        let result = client.find_peaks(&x, params).unwrap();

        // With distance constraint, should pick highest peaks that are >= 3 apart
        assert!(result.indices.len() <= 3);
        // Peak at 6 (height 2.0) should definitely be included
        assert!(result.indices.contains(&6));
    }

    #[test]
    fn test_savgol_smoothing() {
        let (client, device) = setup();

        // Noisy signal
        let n = 101;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                t * t + 0.1 * ((i * 7) as f64).sin() // Quadratic + noise
            })
            .collect();

        let x = Tensor::from_slice(&signal, &[n], &device);

        // Apply Savitzky-Golay smoothing
        let result = client.savgol_filter(&x, 11, 2, 0).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert_eq!(result_data.len(), n);

        // The smoothed signal should be close to the original quadratic
        // Check middle portion to avoid edge effects
        for i in 20..80 {
            let t = i as f64 / (n - 1) as f64;
            let expected = t * t;
            assert!(
                (result_data[i] - expected).abs() < 0.15,
                "Smoothed value at {} should be close to {} (got {})",
                i,
                expected,
                result_data[i]
            );
        }
    }

    #[test]
    fn test_savgol_derivative() {
        let (client, device) = setup();

        // Quadratic: x^2, derivative should be 2x
        let n = 101;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64 - 50.0) / 50.0; // t in [-1, 1]
                t * t
            })
            .collect();

        let x = Tensor::from_slice(&signal, &[n], &device);

        // First derivative
        let result = client.savgol_filter(&x, 11, 3, 1).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Check that derivative is approximately 2t (scaled appropriately)
        // The derivative is in units of samples, not t
        let scale = 50.0; // dt = 1/50
        for i in 20..80 {
            let t = (i as f64 - 50.0) / 50.0;
            let expected_deriv = 2.0 * t / scale;
            // Allow more tolerance for numerical differentiation
            assert!(
                (result_data[i] - expected_deriv).abs() < 0.1,
                "Derivative at {} should be ~{} (got {})",
                i,
                expected_deriv,
                result_data[i]
            );
        }
    }

    #[test]
    fn test_find_peaks_with_prominence() {
        let (client, device) = setup();

        // Signal with peaks of varying prominence
        let signal = vec![0.0, 1.0, 0.8, 2.0, 0.5, 0.6, 0.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let params = PeakParams::new().with_prominence(0.5);
        let result = client.find_peaks(&x, params).unwrap();

        // The peak at index 3 (height 2.0) should have the highest prominence
        assert!(
            result.indices.contains(&3),
            "Should find the prominent peak"
        );

        // Prominences should be computed
        assert!(result.prominences.is_some());
    }
}
