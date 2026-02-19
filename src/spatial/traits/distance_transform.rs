//! Distance transform algorithm traits.
use crate::DType;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Distance metric for distance transforms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DistanceTransformMetric {
    /// Euclidean distance (exact).
    #[default]
    Euclidean,
    /// City block / Manhattan distance.
    CityBlock,
    /// Chessboard / Chebyshev distance.
    Chessboard,
}

/// Algorithmic contract for distance transform operations.
pub trait DistanceTransformAlgorithms<R: Runtime<DType = DType>> {
    /// Compute the distance transform of a binary array.
    ///
    /// For each background pixel (zero), computes the distance to the nearest
    /// foreground pixel (nonzero).
    ///
    /// # Arguments
    ///
    /// * `input` - Binary input tensor (nonzero = foreground)
    /// * `metric` - Distance metric
    ///
    /// # Returns
    ///
    /// Distance tensor with same shape as input. Foreground pixels have distance 0.
    fn distance_transform(
        &self,
        input: &Tensor<R>,
        metric: DistanceTransformMetric,
    ) -> Result<Tensor<R>>;

    /// Compute EDT (Exact Distance Transform) of a binary array.
    ///
    /// Uses the Felzenszwalb & Huttenlocher algorithm for exact Euclidean
    /// distance transform. Each dimension is processed independently.
    ///
    /// # Arguments
    ///
    /// * `input` - Binary input tensor (nonzero = foreground)
    ///
    /// # Returns
    ///
    /// Euclidean distance tensor. Values are actual distances (not squared).
    fn distance_transform_edt(&self, input: &Tensor<R>) -> Result<Tensor<R>>;
}
