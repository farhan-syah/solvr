//! CPU implementation of IIR filter design.

use crate::signal::filter::impl_generic::{
    besselap_impl, bilinear_zpk_impl, buttap_impl, butter_impl, cheb1ap_impl, cheb2ap_impl,
    cheby1_impl, cheby2_impl, ellip_impl, ellipap_impl, lp2bp_zpk_impl, lp2bs_zpk_impl,
    lp2hp_zpk_impl, lp2lp_zpk_impl,
};
use crate::signal::filter::traits::iir_design::{BesselNorm, IirDesignAlgorithms, IirDesignResult};
use crate::signal::filter::types::{AnalogPrototype, FilterOutput, FilterType, ZpkFilter};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};

impl IirDesignAlgorithms<CpuRuntime> for CpuClient {
    fn butter(
        &self,
        order: usize,
        wn: &[f64],
        filter_type: FilterType,
        output: FilterOutput,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<IirDesignResult<CpuRuntime>> {
        butter_impl(self, order, wn, filter_type, output, device)
    }

    fn cheby1(
        &self,
        order: usize,
        rp: f64,
        wn: &[f64],
        filter_type: FilterType,
        output: FilterOutput,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<IirDesignResult<CpuRuntime>> {
        cheby1_impl(self, order, rp, wn, filter_type, output, device)
    }

    fn cheby2(
        &self,
        order: usize,
        rs: f64,
        wn: &[f64],
        filter_type: FilterType,
        output: FilterOutput,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<IirDesignResult<CpuRuntime>> {
        cheby2_impl(self, order, rs, wn, filter_type, output, device)
    }

    fn ellip(
        &self,
        order: usize,
        rp: f64,
        rs: f64,
        wn: &[f64],
        filter_type: FilterType,
        output: FilterOutput,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<IirDesignResult<CpuRuntime>> {
        ellip_impl(self, order, rp, rs, wn, filter_type, output, device)
    }

    fn bessel(
        &self,
        order: usize,
        wn: &[f64],
        filter_type: FilterType,
        output: FilterOutput,
        norm: Option<BesselNorm>,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<IirDesignResult<CpuRuntime>> {
        let norm = norm.unwrap_or_default();
        let proto = besselap_impl(self, order, norm, device)?;
        crate::signal::filter::impl_generic::design_iir_filter(
            self,
            proto,
            wn,
            filter_type,
            output,
            device,
        )
    }

    fn buttap(
        &self,
        order: usize,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<AnalogPrototype<CpuRuntime>> {
        buttap_impl(self, order, device)
    }

    fn cheb1ap(
        &self,
        order: usize,
        rp: f64,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<AnalogPrototype<CpuRuntime>> {
        cheb1ap_impl(self, order, rp, device)
    }

    fn cheb2ap(
        &self,
        order: usize,
        rs: f64,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<AnalogPrototype<CpuRuntime>> {
        cheb2ap_impl(self, order, rs, device)
    }

    fn ellipap(
        &self,
        order: usize,
        rp: f64,
        rs: f64,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<AnalogPrototype<CpuRuntime>> {
        ellipap_impl(self, order, rp, rs, device)
    }

    fn besselap(
        &self,
        order: usize,
        norm: BesselNorm,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<AnalogPrototype<CpuRuntime>> {
        besselap_impl(self, order, norm, device)
    }

    fn bilinear_zpk(
        &self,
        analog: &AnalogPrototype<CpuRuntime>,
        fs: f64,
    ) -> Result<ZpkFilter<CpuRuntime>> {
        bilinear_zpk_impl(self, analog, fs)
    }

    fn lp2lp_zpk(
        &self,
        zpk: &AnalogPrototype<CpuRuntime>,
        wo: f64,
    ) -> Result<AnalogPrototype<CpuRuntime>> {
        lp2lp_zpk_impl(self, zpk, wo)
    }

    fn lp2hp_zpk(
        &self,
        zpk: &AnalogPrototype<CpuRuntime>,
        wo: f64,
    ) -> Result<AnalogPrototype<CpuRuntime>> {
        lp2hp_zpk_impl(self, zpk, wo)
    }

    fn lp2bp_zpk(
        &self,
        zpk: &AnalogPrototype<CpuRuntime>,
        wo: f64,
        bw: f64,
    ) -> Result<AnalogPrototype<CpuRuntime>> {
        lp2bp_zpk_impl(self, zpk, wo, bw)
    }

    fn lp2bs_zpk(
        &self,
        zpk: &AnalogPrototype<CpuRuntime>,
        wo: f64,
        bw: f64,
    ) -> Result<AnalogPrototype<CpuRuntime>> {
        lp2bs_zpk_impl(self, zpk, wo, bw)
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
    fn test_butter_lowpass() {
        let (client, device) = setup();

        let result = client
            .butter(4, &[0.2], FilterType::Lowpass, FilterOutput::Ba, &device)
            .unwrap();

        let tf = result.as_ba().unwrap();
        assert_eq!(tf.b.shape()[0], 5); // 4th order = 5 coefficients
        assert_eq!(tf.a.shape()[0], 5);
    }

    #[test]
    fn test_butter_highpass() {
        let (client, device) = setup();

        let result = client
            .butter(2, &[0.3], FilterType::Highpass, FilterOutput::Sos, &device)
            .unwrap();

        let sos = result.as_sos().unwrap();
        assert_eq!(sos.num_sections(), 1); // 2nd order = 1 section
    }

    #[test]
    fn test_butter_bandpass() {
        let (client, device) = setup();

        let result = client
            .butter(
                2,
                &[0.2, 0.4],
                FilterType::Bandpass,
                FilterOutput::Zpk,
                &device,
            )
            .unwrap();

        let zpk = result.as_zpk().unwrap();
        // Bandpass doubles the order
        assert_eq!(zpk.num_poles(), 4);
    }

    #[test]
    fn test_cheby1() {
        let (client, device) = setup();

        let result = client
            .cheby1(
                3,
                1.0,
                &[0.25],
                FilterType::Lowpass,
                FilterOutput::Ba,
                &device,
            )
            .unwrap();

        let tf = result.as_ba().unwrap();
        assert_eq!(tf.order(), 3);
    }

    #[test]
    fn test_cheby2() {
        let (client, device) = setup();

        let result = client
            .cheby2(
                3,
                40.0,
                &[0.25],
                FilterType::Lowpass,
                FilterOutput::Sos,
                &device,
            )
            .unwrap();

        let sos = result.as_sos().unwrap();
        assert!(sos.num_sections() >= 1);
    }

    #[test]
    fn test_buttap() {
        let (client, device) = setup();

        let proto = client.buttap(4, &device).unwrap();

        assert_eq!(proto.poles_real.shape()[0], 4);
        assert_eq!(proto.zeros_real.shape()[0], 0);

        // All poles should have negative real part
        let poles_re: Vec<f64> = proto.poles_real.to_vec();
        for &p in &poles_re {
            assert!(p < 0.0);
        }
    }
}
