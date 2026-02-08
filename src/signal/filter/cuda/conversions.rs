//! CUDA implementation of filter conversions.
//!
//! Note: Some conversions (sos2tf, zpk2sos, sos2zpk) involve inherently sequential
//! algorithms with tiny data sizes. These are NOT implemented for CUDA because:
//! 1. GPU transfer overhead exceeds computation time
//! 2. Design-time operations that only run once
//! 3. Users should use CpuClient for filter design, then transfer to GPU for filtering

use crate::signal::filter::impl_generic::{tf2zpk_impl, zpk2tf_impl};
use crate::signal::filter::traits::conversions::{FilterConversions, SosPairing};
use crate::signal::filter::types::{SosFilter, TransferFunction, ZpkFilter};
use numr::error::{Error, Result};
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
        _tf: &TransferFunction<CudaRuntime>,
        _pairing: Option<SosPairing>,
    ) -> Result<SosFilter<CudaRuntime>> {
        Err(Error::backend_limitation(
            "cuda",
            "tf2sos",
            "Filter conversion is CPU-only. Use CpuClient for filter design, then transfer to GPU.",
        ))
    }

    fn sos2tf(&self, _sos: &SosFilter<CudaRuntime>) -> Result<TransferFunction<CudaRuntime>> {
        Err(Error::backend_limitation(
            "cuda",
            "sos2tf",
            "Filter conversion is CPU-only. Use CpuClient for filter design.",
        ))
    }

    fn zpk2sos(
        &self,
        _zpk: &ZpkFilter<CudaRuntime>,
        _pairing: Option<SosPairing>,
    ) -> Result<SosFilter<CudaRuntime>> {
        Err(Error::backend_limitation(
            "cuda",
            "zpk2sos",
            "Filter conversion is CPU-only. Use CpuClient for filter design.",
        ))
    }

    fn sos2zpk(&self, _sos: &SosFilter<CudaRuntime>) -> Result<ZpkFilter<CudaRuntime>> {
        Err(Error::backend_limitation(
            "cuda",
            "sos2zpk",
            "Filter conversion is CPU-only. Use CpuClient for filter design.",
        ))
    }
}
