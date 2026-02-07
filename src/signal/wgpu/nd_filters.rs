//! WebGPU implementation of N-dimensional filter algorithms.

use crate::signal::impl_generic::{
    gaussian_filter_impl, maximum_filter_impl, minimum_filter_impl, percentile_filter_impl,
    uniform_filter_impl,
};
use crate::signal::traits::nd_filters::{BoundaryMode, NdFilterAlgorithms};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl NdFilterAlgorithms<WgpuRuntime> for WgpuClient {
    fn gaussian_filter(
        &self,
        input: &Tensor<WgpuRuntime>,
        sigma: &[f64],
        order: &[usize],
        mode: BoundaryMode,
        truncate: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        gaussian_filter_impl(self, input, sigma, order, mode, truncate)
    }

    fn uniform_filter(
        &self,
        input: &Tensor<WgpuRuntime>,
        size: &[usize],
        mode: BoundaryMode,
    ) -> Result<Tensor<WgpuRuntime>> {
        uniform_filter_impl(self, input, size, mode)
    }

    fn minimum_filter(
        &self,
        input: &Tensor<WgpuRuntime>,
        size: &[usize],
        mode: BoundaryMode,
    ) -> Result<Tensor<WgpuRuntime>> {
        minimum_filter_impl(self, input, size, mode)
    }

    fn maximum_filter(
        &self,
        input: &Tensor<WgpuRuntime>,
        size: &[usize],
        mode: BoundaryMode,
    ) -> Result<Tensor<WgpuRuntime>> {
        maximum_filter_impl(self, input, size, mode)
    }

    fn percentile_filter(
        &self,
        input: &Tensor<WgpuRuntime>,
        percentile: f64,
        size: &[usize],
        mode: BoundaryMode,
    ) -> Result<Tensor<WgpuRuntime>> {
        percentile_filter_impl(self, input, percentile, size, mode)
    }
}
