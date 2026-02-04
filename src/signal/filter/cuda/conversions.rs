//! CUDA implementation of filter conversions.

use crate::signal::filter::impl_generic::{
    sos2tf_impl, sos2zpk_impl, tf2sos_impl, tf2zpk_impl, zpk2sos_impl, zpk2tf_impl,
};
use crate::signal::filter::traits::conversions::{FilterConversions, SosPairing};
use crate::signal::filter::types::{SosFilter, TransferFunction, ZpkFilter};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};

impl FilterConversions<CudaRuntime> for CudaClient {
    fn tf2zpk(&self, tf: &TransferFunction<CudaRuntime>) -> Result<ZpkFilter<CudaRuntime>> {
        tf2zpk_impl(self, tf)
    }

    fn zpk2tf(&self, zpk: &ZpkFilter<CudaRuntime>) -> Result<TransferFunction<CudaRuntime>> {
        zpk2tf_impl(self, zpk)
    }

    fn tf2sos(
        &self,
        tf: &TransferFunction<CudaRuntime>,
        pairing: Option<SosPairing>,
    ) -> Result<SosFilter<CudaRuntime>> {
        tf2sos_impl(self, tf, pairing)
    }

    fn sos2tf(&self, sos: &SosFilter<CudaRuntime>) -> Result<TransferFunction<CudaRuntime>> {
        sos2tf_impl(self, sos)
    }

    fn zpk2sos(
        &self,
        zpk: &ZpkFilter<CudaRuntime>,
        pairing: Option<SosPairing>,
    ) -> Result<SosFilter<CudaRuntime>> {
        zpk2sos_impl(self, zpk, pairing)
    }

    fn sos2zpk(&self, sos: &SosFilter<CudaRuntime>) -> Result<ZpkFilter<CudaRuntime>> {
        sos2zpk_impl(self, sos)
    }
}
