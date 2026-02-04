//! CUDA implementation of FIR filter design.

use crate::signal::filter::impl_generic::{firwin_impl, firwin2_impl, minimum_phase_impl};
use crate::signal::filter::traits::fir_design::{FirDesignAlgorithms, FirWindow};
use crate::signal::filter::types::FilterType;
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl FirDesignAlgorithms<CudaRuntime> for CudaClient {
    fn firwin(
        &self,
        numtaps: usize,
        cutoff: &[f64],
        filter_type: FilterType,
        window: FirWindow,
        scale: bool,
        device: &<CudaRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<Tensor<CudaRuntime>> {
        firwin_impl(self, numtaps, cutoff, filter_type, window, scale, device)
    }

    fn firwin2(
        &self,
        numtaps: usize,
        freq: &[f64],
        gain: &[f64],
        antisymmetric: bool,
        window: FirWindow,
        device: &<CudaRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<Tensor<CudaRuntime>> {
        firwin2_impl(self, numtaps, freq, gain, antisymmetric, window, device)
    }

    fn minimum_phase(&self, h: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        minimum_phase_impl(self, h)
    }
}
