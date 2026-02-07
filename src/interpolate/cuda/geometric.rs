//! CUDA implementation of geometric transform algorithms.

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::geometric::{
    affine_transform_impl, map_coordinates_impl, rotate_impl, shift_impl, zoom_impl,
};
use crate::interpolate::traits::geometric::{GeometricTransformAlgorithms, InterpolationOrder};
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl GeometricTransformAlgorithms<CudaRuntime> for CudaClient {
    fn map_coordinates(
        &self,
        input: &Tensor<CudaRuntime>,
        coordinates: &Tensor<CudaRuntime>,
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        map_coordinates_impl(self, input, coordinates, order, cval)
    }

    fn affine_transform(
        &self,
        input: &Tensor<CudaRuntime>,
        matrix: &Tensor<CudaRuntime>,
        offset: &Tensor<CudaRuntime>,
        output_shape: Option<&[usize]>,
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        affine_transform_impl(self, input, matrix, offset, output_shape, order, cval)
    }

    fn zoom(
        &self,
        input: &Tensor<CudaRuntime>,
        zoom: &[f64],
        order: InterpolationOrder,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        zoom_impl(self, input, zoom, order)
    }

    fn rotate(
        &self,
        input: &Tensor<CudaRuntime>,
        angle: f64,
        axes: (usize, usize),
        reshape: bool,
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        rotate_impl(self, input, angle, axes, reshape, order, cval)
    }

    fn shift(
        &self,
        input: &Tensor<CudaRuntime>,
        shift: &[f64],
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<CudaRuntime>> {
        shift_impl(self, input, shift, order, cval)
    }
}
