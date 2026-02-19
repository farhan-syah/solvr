//! Halfspace intersection trait.
//!
//! Computes the convex polytope formed by the intersection of halfspaces.
//! Each halfspace is defined as n·x + b ≤ 0.
use crate::DType;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Halfspace intersection result.
#[derive(Debug, Clone)]
pub struct HalfspaceIntersection<R: Runtime<DType = DType>> {
    /// Input halfspaces `[m, d+1]`. Each row is `[n_1, ..., n_d, b]`
    /// representing the halfspace n·x + b ≤ 0.
    pub halfspaces: Tensor<R>,

    /// Vertices of the intersection polytope `[n_vertices, d]`.
    pub intersections: Tensor<R>,

    /// Dual points used in computation `[m, d]`.
    pub dual_points: Tensor<R>,

    /// Interior point `[d]`.
    pub interior_point: Tensor<R>,
}

/// Algorithmic contract for halfspace intersection.
pub trait HalfspaceIntersectionAlgorithms<R: Runtime<DType = DType>> {
    /// Compute the intersection of halfspaces.
    ///
    /// # Arguments
    ///
    /// * `halfspaces` - `[m, d+1]` where each row is `[n_1,...,n_d, b]` for n·x + b ≤ 0
    /// * `interior_point` - `[d]` a point strictly inside all halfspaces
    ///
    /// # Returns
    ///
    /// `HalfspaceIntersection` with the vertices of the intersection polytope.
    fn halfspace_intersection(
        &self,
        halfspaces: &Tensor<R>,
        interior_point: &Tensor<R>,
    ) -> Result<HalfspaceIntersection<R>>;
}
