//! CUDA implementation of FIR filter design.
//!
//! Note: FIR filter design functions are NOT implemented for CUDA because:
//! 1. These are design-time operations (run once), not runtime (per-sample)
//! 2. FIR tap counts are typically 31-255 (tiny data)
//! 3. GPU transfer overhead far exceeds computation time
//!
//! Users should use CpuClient for filter design, then transfer the coefficients to GPU.

use crate::signal::filter::traits::fir_design::{FirDesignAlgorithms, FirWindow};
use crate::signal::filter::types::FilterType;
use numr::error::{Error, Result};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl FirDesignAlgorithms<CudaRuntime> for CudaClient {
    fn firwin(
        &self,
        _numtaps: usize,
        _cutoff: &[f64],
        _filter_type: FilterType,
        _window: FirWindow,
        _scale: bool,
        _device: &<CudaRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<Tensor<CudaRuntime>> {
        Err(Error::UnsupportedOperation {
            operation: "firwin is CPU-only. Use CpuClient for filter design, then transfer coefficients to GPU.".to_string(),
        })
    }

    fn firwin2(
        &self,
        _numtaps: usize,
        _freq: &[f64],
        _gain: &[f64],
        _antisymmetric: bool,
        _window: FirWindow,
        _device: &<CudaRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<Tensor<CudaRuntime>> {
        Err(Error::UnsupportedOperation {
            operation: "firwin2 is CPU-only. Use CpuClient for filter design, then transfer coefficients to GPU.".to_string(),
        })
    }

    fn minimum_phase(&self, _h: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        Err(Error::UnsupportedOperation {
            operation: "minimum_phase is CPU-only. Use CpuClient for filter design.".to_string(),
        })
    }
}
