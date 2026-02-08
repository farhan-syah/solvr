//! WebGPU implementation of FIR filter design.
//!
//! Note: FIR filter design functions are NOT implemented for WebGPU because:
//! 1. These are design-time operations (run once), not runtime (per-sample)
//! 2. FIR tap counts are typically 31-255 (tiny data)
//! 3. GPU transfer overhead far exceeds computation time
//!
//! Users should use CpuClient for filter design, then transfer the coefficients to GPU.

use crate::signal::filter::traits::fir_design::{FirDesignAlgorithms, FirWindow};
use crate::signal::filter::types::FilterType;
use numr::error::{Error, Result};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl FirDesignAlgorithms<WgpuRuntime> for WgpuClient {
    fn firwin(
        &self,
        _numtaps: usize,
        _cutoff: &[f64],
        _filter_type: FilterType,
        _window: FirWindow,
        _scale: bool,
        _device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<Tensor<WgpuRuntime>> {
        Err(Error::backend_limitation(
            "wgpu",
            "firwin",
            "FIR filter design is CPU-only. Use CpuClient for filter design, then transfer coefficients to GPU.",
        ))
    }

    fn firwin2(
        &self,
        _numtaps: usize,
        _freq: &[f64],
        _gain: &[f64],
        _antisymmetric: bool,
        _window: FirWindow,
        _device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<Tensor<WgpuRuntime>> {
        Err(Error::backend_limitation(
            "wgpu",
            "firwin2",
            "FIR filter design is CPU-only. Use CpuClient for filter design, then transfer coefficients to GPU.",
        ))
    }

    fn minimum_phase(&self, _h: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        Err(Error::backend_limitation(
            "wgpu",
            "minimum_phase",
            "FIR filter design is CPU-only. Use CpuClient for filter design.",
        ))
    }
}
