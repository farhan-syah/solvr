//! Edge detection algorithm traits.
//!
//! Provides algorithms for detecting edges in N-dimensional arrays using
//! derivative-based filters (Sobel, Prewitt, Laplace).

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Algorithmic contract for edge detection operations.
///
/// All backends implementing edge detection MUST implement this trait using
/// the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait EdgeDetectionAlgorithms<R: Runtime> {
    /// Apply Sobel filter along the given axis.
    ///
    /// Computes an approximation of the gradient along one axis using a Sobel
    /// operator. For 2D arrays, this is a 3x3 convolution kernel.
    /// For N-D, applies separable Sobel along the specified axis.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor (at least 2D)
    /// * `axis` - Axis along which to compute the gradient
    ///
    /// # Returns
    ///
    /// Gradient tensor with same shape as input.
    fn sobel(&self, input: &Tensor<R>, axis: usize) -> Result<Tensor<R>>;

    /// Apply Prewitt filter along the given axis.
    ///
    /// Similar to Sobel but uses equal weighting instead of Gaussian-weighted
    /// smoothing in the orthogonal direction.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor (at least 2D)
    /// * `axis` - Axis along which to compute the gradient
    fn prewitt(&self, input: &Tensor<R>, axis: usize) -> Result<Tensor<R>>;

    /// Apply Laplacian filter.
    ///
    /// Computes the Laplacian (sum of second derivatives) of the input using
    /// second-difference operators along each axis.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor (any dimensionality >= 1)
    fn laplace(&self, input: &Tensor<R>) -> Result<Tensor<R>>;

    /// Apply Gaussian then Laplacian (Laplacian of Gaussian / LoG).
    ///
    /// First smooths with a Gaussian filter, then applies the Laplacian.
    /// Useful for blob detection.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `sigma` - Standard deviation for Gaussian smoothing (applied to all axes)
    fn gaussian_laplace(&self, input: &Tensor<R>, sigma: f64) -> Result<Tensor<R>>;

    /// Compute the gradient magnitude after Gaussian smoothing.
    ///
    /// Smooths input with Gaussian, computes gradient along each axis,
    /// returns sqrt(sum of squared gradients).
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `sigma` - Standard deviation for Gaussian smoothing (applied to all axes)
    fn gaussian_gradient_magnitude(&self, input: &Tensor<R>, sigma: f64) -> Result<Tensor<R>>;
}
