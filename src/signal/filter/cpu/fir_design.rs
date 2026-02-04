//! CPU implementation of FIR filter design.

use crate::signal::filter::impl_generic::{firwin_impl, firwin2_impl, minimum_phase_impl};
use crate::signal::filter::traits::fir_design::{FirDesignAlgorithms, FirWindow};
use crate::signal::filter::types::FilterType;
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl FirDesignAlgorithms<CpuRuntime> for CpuClient {
    fn firwin(
        &self,
        numtaps: usize,
        cutoff: &[f64],
        filter_type: FilterType,
        window: FirWindow,
        scale: bool,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        firwin_impl(self, numtaps, cutoff, filter_type, window, scale, device)
    }

    fn firwin2(
        &self,
        numtaps: usize,
        freq: &[f64],
        gain: &[f64],
        antisymmetric: bool,
        window: FirWindow,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        firwin2_impl(self, numtaps, freq, gain, antisymmetric, window, device)
    }

    fn minimum_phase(&self, h: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        minimum_phase_impl(self, h)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_firwin_lowpass() {
        let (client, device) = setup();

        let h = client
            .firwin(
                51,
                &[0.2],
                FilterType::Lowpass,
                FirWindow::Hamming,
                true,
                &device,
            )
            .unwrap();

        assert_eq!(h.shape(), &[51]);

        // Verify basic filter properties
        let h_data: Vec<f64> = h.to_vec();

        // Center value should be positive and largest
        let center = 25;
        assert!(h_data[center] > 0.0, "Center tap should be positive");

        // Taps should decay away from center
        assert!(h_data[center] > h_data[center - 1].abs());
        assert!(h_data[center] > h_data[center + 1].abs());

        // Sum of coefficients should be approximately 1 (DC gain)
        let sum: f64 = h_data.iter().sum();
        assert!((sum - 1.0).abs() < 0.1, "DC gain should be ~1, got {}", sum);
    }

    #[test]
    fn test_firwin_highpass() {
        let (client, device) = setup();

        // Highpass requires odd number of taps
        let h = client
            .firwin(
                51,
                &[0.3],
                FilterType::Highpass,
                FirWindow::Hann,
                true,
                &device,
            )
            .unwrap();

        assert_eq!(h.shape(), &[51]);
    }

    #[test]
    fn test_firwin_bandpass() {
        let (client, device) = setup();

        let h = client
            .firwin(
                101,
                &[0.2, 0.4],
                FilterType::Bandpass,
                FirWindow::Blackman,
                true,
                &device,
            )
            .unwrap();

        assert_eq!(h.shape(), &[101]);
    }

    #[test]
    fn test_firwin_kaiser() {
        let (client, device) = setup();

        let h = client
            .firwin(
                65,
                &[0.25],
                FilterType::Lowpass,
                FirWindow::Kaiser(5.0),
                true,
                &device,
            )
            .unwrap();

        assert_eq!(h.shape(), &[65]);
    }

    #[test]
    fn test_firwin2() {
        let (client, device) = setup();

        // Design a filter with specific frequency response
        let freq = vec![0.0, 0.2, 0.3, 1.0];
        let gain = vec![1.0, 1.0, 0.0, 0.0];

        let h = client
            .firwin2(51, &freq, &gain, false, FirWindow::Hamming, &device)
            .unwrap();

        assert_eq!(h.shape(), &[51]);
    }
}
