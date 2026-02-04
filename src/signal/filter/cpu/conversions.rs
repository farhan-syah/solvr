//! CPU implementation of filter conversions.

use crate::signal::filter::impl_generic::{
    sos2tf_impl, sos2zpk_impl, tf2sos_impl, tf2zpk_impl, zpk2sos_impl, zpk2tf_impl,
};
use crate::signal::filter::traits::conversions::{FilterConversions, SosPairing};
use crate::signal::filter::types::{SosFilter, TransferFunction, ZpkFilter};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};

impl FilterConversions<CpuRuntime> for CpuClient {
    fn tf2zpk(&self, tf: &TransferFunction<CpuRuntime>) -> Result<ZpkFilter<CpuRuntime>> {
        tf2zpk_impl(self, tf)
    }

    fn zpk2tf(&self, zpk: &ZpkFilter<CpuRuntime>) -> Result<TransferFunction<CpuRuntime>> {
        zpk2tf_impl(self, zpk)
    }

    fn tf2sos(
        &self,
        tf: &TransferFunction<CpuRuntime>,
        pairing: Option<SosPairing>,
    ) -> Result<SosFilter<CpuRuntime>> {
        tf2sos_impl(self, tf, pairing)
    }

    fn sos2tf(&self, sos: &SosFilter<CpuRuntime>) -> Result<TransferFunction<CpuRuntime>> {
        sos2tf_impl(self, sos)
    }

    fn zpk2sos(
        &self,
        zpk: &ZpkFilter<CpuRuntime>,
        pairing: Option<SosPairing>,
    ) -> Result<SosFilter<CpuRuntime>> {
        zpk2sos_impl(self, zpk, pairing)
    }

    fn sos2zpk(&self, sos: &SosFilter<CpuRuntime>) -> Result<ZpkFilter<CpuRuntime>> {
        sos2zpk_impl(self, sos)
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
    fn test_tf2zpk_simple() {
        let (client, device) = setup();

        // Simple lowpass: H(z) = 1 / (1 - 0.5z^-1)
        // Pole at z = 0.5
        use numr::tensor::Tensor;

        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1], &device);
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -0.5], &[2], &device);
        let tf = TransferFunction::new(b, a);

        let zpk = client.tf2zpk(&tf).unwrap();

        assert_eq!(zpk.num_zeros(), 0);
        assert_eq!(zpk.num_poles(), 1);

        let poles_re: Vec<f64> = zpk.poles_real.to_vec();
        assert!((poles_re[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_zpk2tf_roundtrip() {
        let (client, device) = setup();
        use numr::tensor::Tensor;

        // Create a simple filter
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -1.0], &[2], &device);
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, -0.9], &[2], &device);
        let tf_orig = TransferFunction::new(b, a);

        // Convert to ZPK and back
        let zpk = client.tf2zpk(&tf_orig).unwrap();
        let tf_back = client.zpk2tf(&zpk).unwrap();

        let b_orig: Vec<f64> = tf_orig.b.to_vec();
        let b_back: Vec<f64> = tf_back.b.to_vec();

        // Normalize for comparison
        let scale = b_orig[0] / b_back[0];
        for i in 0..b_orig.len() {
            assert!((b_orig[i] - b_back[i] * scale).abs() < 1e-6);
        }
    }

    #[test]
    fn test_tf2sos() {
        let (client, device) = setup();
        use numr::tensor::Tensor;

        // 4th order filter (2 biquad sections)
        let b = Tensor::<CpuRuntime>::from_slice(&[0.1, 0.4, 0.6, 0.4, 0.1], &[5], &device);
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0, -0.8, 0.64, -0.256, 0.0625], &[5], &device);
        let tf = TransferFunction::new(b, a);

        let sos = client.tf2sos(&tf, None).unwrap();

        // Should have 2 sections for 4th order
        assert_eq!(sos.num_sections(), 2);
        assert_eq!(sos.sections.shape(), &[2, 6]);
    }
}
