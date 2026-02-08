//! WebGPU implementation of filter conversions.
//!
//! Note: Some conversions (sos2tf, zpk2sos, sos2zpk) involve inherently sequential
//! algorithms with tiny data sizes. These are NOT implemented for WebGPU because:
//! 1. GPU transfer overhead exceeds computation time
//! 2. Design-time operations that only run once
//! 3. Users should use CpuClient for filter design, then transfer to GPU for filtering

use crate::signal::filter::impl_generic::{tf2zpk_impl, zpk2tf_impl};
use crate::signal::filter::traits::conversions::{FilterConversions, SosPairing};
use crate::signal::filter::types::{SosFilter, TransferFunction, ZpkFilter};
use numr::error::{Error, Result};
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
        _tf: &TransferFunction<WgpuRuntime>,
        _pairing: Option<SosPairing>,
    ) -> Result<SosFilter<WgpuRuntime>> {
        Err(Error::backend_limitation(
            "wgpu",
            "tf2sos",
            "Filter conversion is CPU-only. Use CpuClient for filter design, then transfer to GPU.",
        ))
    }

    fn sos2tf(&self, _sos: &SosFilter<WgpuRuntime>) -> Result<TransferFunction<WgpuRuntime>> {
        Err(Error::backend_limitation(
            "wgpu",
            "sos2tf",
            "Filter conversion is CPU-only. Use CpuClient for filter design.",
        ))
    }

    fn zpk2sos(
        &self,
        _zpk: &ZpkFilter<WgpuRuntime>,
        _pairing: Option<SosPairing>,
    ) -> Result<SosFilter<WgpuRuntime>> {
        Err(Error::backend_limitation(
            "wgpu",
            "zpk2sos",
            "Filter conversion is CPU-only. Use CpuClient for filter design.",
        ))
    }

    fn sos2zpk(&self, _sos: &SosFilter<WgpuRuntime>) -> Result<ZpkFilter<WgpuRuntime>> {
        Err(Error::backend_limitation(
            "wgpu",
            "sos2zpk",
            "Filter conversion is CPU-only. Use CpuClient for filter design.",
        ))
    }
}
