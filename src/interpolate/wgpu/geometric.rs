//! WebGPU implementation of geometric transform algorithms.

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::geometric::{
    affine_transform_impl, map_coordinates_impl, rotate_impl, shift_impl, zoom_impl,
};
use crate::interpolate::traits::geometric::{GeometricTransformAlgorithms, InterpolationOrder};
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl GeometricTransformAlgorithms<WgpuRuntime> for WgpuClient {
    fn map_coordinates(
        &self,
        input: &Tensor<WgpuRuntime>,
        coordinates: &Tensor<WgpuRuntime>,
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        map_coordinates_impl(self, input, coordinates, order, cval)
    }

    fn affine_transform(
        &self,
        input: &Tensor<WgpuRuntime>,
        matrix: &Tensor<WgpuRuntime>,
        offset: &Tensor<WgpuRuntime>,
        output_shape: Option<&[usize]>,
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        affine_transform_impl(self, input, matrix, offset, output_shape, order, cval)
    }

    fn zoom(
        &self,
        input: &Tensor<WgpuRuntime>,
        zoom: &[f64],
        order: InterpolationOrder,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        zoom_impl(self, input, zoom, order)
    }

    fn rotate(
        &self,
        input: &Tensor<WgpuRuntime>,
        angle: f64,
        axes: (usize, usize),
        reshape: bool,
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        rotate_impl(self, input, angle, axes, reshape, order, cval)
    }

    fn shift(
        &self,
        input: &Tensor<WgpuRuntime>,
        shift: &[f64],
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<WgpuRuntime>> {
        shift_impl(self, input, shift, order, cval)
    }
}
