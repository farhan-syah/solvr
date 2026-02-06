use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::scattered::griddata_impl;
use crate::interpolate::traits::scattered::{ScatteredInterpAlgorithms, ScatteredMethod};
use numr::ops::{CompareOps, MatmulOps, ScalarOps, ShapeOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + MatmulOps<R> + ShapeOps<R> + RuntimeClient<R>,
> ScatteredInterpAlgorithms<R> for C
{
    fn griddata(
        &self,
        points: &Tensor<R>,
        values: &Tensor<R>,
        xi: &Tensor<R>,
        method: ScatteredMethod,
    ) -> InterpolateResult<Tensor<R>> {
        griddata_impl(self, points, values, xi, method)
    }
}
