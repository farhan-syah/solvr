//! WebGPU implementation of IIR filter design.

use crate::signal::filter::impl_generic::{
    besselap_impl, bilinear_zpk_impl, buttap_impl, butter_impl, cheb1ap_impl, cheb2ap_impl,
    cheby1_impl, cheby2_impl, ellip_impl, ellipap_impl, lp2bp_zpk_impl, lp2bs_zpk_impl,
    lp2hp_zpk_impl, lp2lp_zpk_impl,
};
use crate::signal::filter::traits::iir_design::{BesselNorm, IirDesignAlgorithms, IirDesignResult};
use crate::signal::filter::types::{AnalogPrototype, FilterOutput, FilterType, ZpkFilter};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};

impl IirDesignAlgorithms<WgpuRuntime> for WgpuClient {
    fn butter(
        &self,
        order: usize,
        wn: &[f64],
        filter_type: FilterType,
        output: FilterOutput,
        device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<IirDesignResult<WgpuRuntime>> {
        butter_impl(self, order, wn, filter_type, output, device)
    }

    fn cheby1(
        &self,
        order: usize,
        rp: f64,
        wn: &[f64],
        filter_type: FilterType,
        output: FilterOutput,
        device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<IirDesignResult<WgpuRuntime>> {
        cheby1_impl(self, order, rp, wn, filter_type, output, device)
    }

    fn cheby2(
        &self,
        order: usize,
        rs: f64,
        wn: &[f64],
        filter_type: FilterType,
        output: FilterOutput,
        device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<IirDesignResult<WgpuRuntime>> {
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
        device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<IirDesignResult<WgpuRuntime>> {
        ellip_impl(self, order, rp, rs, wn, filter_type, output, device)
    }

    fn bessel(
        &self,
        order: usize,
        wn: &[f64],
        filter_type: FilterType,
        output: FilterOutput,
        norm: Option<BesselNorm>,
        device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<IirDesignResult<WgpuRuntime>> {
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
        device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<AnalogPrototype<WgpuRuntime>> {
        buttap_impl(self, order, device)
    }

    fn cheb1ap(
        &self,
        order: usize,
        rp: f64,
        device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<AnalogPrototype<WgpuRuntime>> {
        cheb1ap_impl(self, order, rp, device)
    }

    fn cheb2ap(
        &self,
        order: usize,
        rs: f64,
        device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<AnalogPrototype<WgpuRuntime>> {
        cheb2ap_impl(self, order, rs, device)
    }

    fn ellipap(
        &self,
        order: usize,
        rp: f64,
        rs: f64,
        device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<AnalogPrototype<WgpuRuntime>> {
        ellipap_impl(self, order, rp, rs, device)
    }

    fn besselap(
        &self,
        order: usize,
        norm: BesselNorm,
        device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<AnalogPrototype<WgpuRuntime>> {
        besselap_impl(self, order, norm, device)
    }

    fn bilinear_zpk(
        &self,
        analog: &AnalogPrototype<WgpuRuntime>,
        fs: f64,
    ) -> Result<ZpkFilter<WgpuRuntime>> {
        bilinear_zpk_impl(self, analog, fs)
    }

    fn lp2lp_zpk(
        &self,
        zpk: &AnalogPrototype<WgpuRuntime>,
        wo: f64,
    ) -> Result<AnalogPrototype<WgpuRuntime>> {
        lp2lp_zpk_impl(self, zpk, wo)
    }

    fn lp2hp_zpk(
        &self,
        zpk: &AnalogPrototype<WgpuRuntime>,
        wo: f64,
    ) -> Result<AnalogPrototype<WgpuRuntime>> {
        lp2hp_zpk_impl(self, zpk, wo)
    }

    fn lp2bp_zpk(
        &self,
        zpk: &AnalogPrototype<WgpuRuntime>,
        wo: f64,
        bw: f64,
    ) -> Result<AnalogPrototype<WgpuRuntime>> {
        lp2bp_zpk_impl(self, zpk, wo, bw)
    }

    fn lp2bs_zpk(
        &self,
        zpk: &AnalogPrototype<WgpuRuntime>,
        wo: f64,
        bw: f64,
    ) -> Result<AnalogPrototype<WgpuRuntime>> {
        lp2bs_zpk_impl(self, zpk, wo, bw)
    }
}
