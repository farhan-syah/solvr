use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::bspline::{
    bspline_derivative_impl, bspline_evaluate_impl, bspline_integrate_impl, make_interp_spline_impl,
};
use crate::interpolate::traits::bspline::{BSpline, BSplineAlgorithms, BSplineBoundary};
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::ops::{CompareOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

impl<
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + CompareOps<R> + LinearAlgebraAlgorithms<R> + RuntimeClient<R>,
> BSplineAlgorithms<R> for C
{
    fn make_interp_spline(
        &self,
        x: &Tensor<R>,
        y: &Tensor<R>,
        degree: usize,
        boundary: &BSplineBoundary,
    ) -> InterpolateResult<BSpline<R>> {
        make_interp_spline_impl(self, x, y, degree, boundary)
    }

    fn bspline_evaluate(
        &self,
        spline: &BSpline<R>,
        x_new: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        bspline_evaluate_impl(self, spline, x_new)
    }

    fn bspline_derivative(
        &self,
        spline: &BSpline<R>,
        x_new: &Tensor<R>,
        order: usize,
    ) -> InterpolateResult<Tensor<R>> {
        bspline_derivative_impl(self, spline, x_new, order)
    }

    fn bspline_integrate(
        &self,
        spline: &BSpline<R>,
        a: f64,
        b: f64,
    ) -> InterpolateResult<Tensor<R>> {
        bspline_integrate_impl(self, spline, a, b)
    }
}
