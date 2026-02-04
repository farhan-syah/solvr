//! WebGPU implementation of filter conversions.

use crate::signal::filter::impl_generic::{
    sos2tf_impl, sos2zpk_impl, tf2sos_impl, tf2zpk_impl, zpk2sos_impl, zpk2tf_impl,
};
use crate::signal::filter::traits::conversions::{FilterConversions, SosPairing};
use crate::signal::filter::types::{SosFilter, TransferFunction, ZpkFilter};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};

impl FilterConversions<WgpuRuntime> for WgpuClient {
    fn tf2zpk(&self, tf: &TransferFunction<WgpuRuntime>) -> Result<ZpkFilter<WgpuRuntime>> {
        tf2zpk_impl(self, tf)
    }

    fn zpk2tf(&self, zpk: &ZpkFilter<WgpuRuntime>) -> Result<TransferFunction<WgpuRuntime>> {
        zpk2tf_impl(self, zpk)
    }

    fn tf2sos(
        &self,
        tf: &TransferFunction<WgpuRuntime>,
        pairing: Option<SosPairing>,
    ) -> Result<SosFilter<WgpuRuntime>> {
        tf2sos_impl(self, tf, pairing)
    }

    fn sos2tf(&self, sos: &SosFilter<WgpuRuntime>) -> Result<TransferFunction<WgpuRuntime>> {
        sos2tf_impl(self, sos)
    }

    fn zpk2sos(
        &self,
        zpk: &ZpkFilter<WgpuRuntime>,
        pairing: Option<SosPairing>,
    ) -> Result<SosFilter<WgpuRuntime>> {
        zpk2sos_impl(self, zpk, pairing)
    }

    fn sos2zpk(&self, sos: &SosFilter<WgpuRuntime>) -> Result<ZpkFilter<WgpuRuntime>> {
        sos2zpk_impl(self, sos)
    }
}
