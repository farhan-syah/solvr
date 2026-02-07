//! N-dimensional filter algorithm traits.
//!
//! Provides algorithms for N-dimensional filtering of arrays (Gaussian, uniform,
//! minimum, maximum, percentile filters with configurable boundary handling).

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Boundary handling mode for N-D filters.
///
/// Determines how values outside the array boundaries are computed.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum BoundaryMode {
    /// Reflect: d c b a | a b c d | d c b a (half-sample symmetric)
    #[default]
    Reflect,
    /// Pad with a constant value: k k k k | a b c d | k k k k
    Constant(f64),
    /// Nearest edge value: a a a a | a b c d | d d d d
    Nearest,
    /// Mirror: d c b | a b c d | c b a (whole-sample symmetric)
    Mirror,
    /// Wrap (periodic): a b c d | a b c d | a b c d
    Wrap,
}

/// Algorithmic contract for N-dimensional filter operations.
///
/// All backends implementing N-D filtering MUST implement this trait using
/// the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait NdFilterAlgorithms<R: Runtime> {
    /// Apply a Gaussian filter to an N-dimensional array.
    ///
    /// Implements separable Gaussian filtering: applies 1D Gaussian convolution
    /// along each axis independently.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor (any dimensionality)
    /// * `sigma` - Standard deviation for each axis. If shorter than ndim,
    ///   the last value is repeated for remaining axes.
    /// * `order` - Derivative order for each axis (0 = smoothing, 1 = first derivative, etc.).
    ///   If empty, defaults to 0 for all axes.
    /// * `mode` - Boundary handling mode
    /// * `truncate` - Truncate the filter at this many standard deviations (default: 4.0)
    ///
    /// # Returns
    ///
    /// Filtered tensor with same shape as input.
    fn gaussian_filter(
        &self,
        input: &Tensor<R>,
        sigma: &[f64],
        order: &[usize],
        mode: BoundaryMode,
        truncate: f64,
    ) -> Result<Tensor<R>>;

    /// Apply a uniform (box) filter to an N-dimensional array.
    ///
    /// Implements separable box filtering: applies 1D uniform convolution
    /// along each axis independently.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor (any dimensionality)
    /// * `size` - Filter size for each axis. If shorter than ndim,
    ///   the last value is repeated for remaining axes.
    /// * `mode` - Boundary handling mode
    ///
    /// # Returns
    ///
    /// Filtered tensor with same shape as input.
    fn uniform_filter(
        &self,
        input: &Tensor<R>,
        size: &[usize],
        mode: BoundaryMode,
    ) -> Result<Tensor<R>>;

    /// Apply a minimum filter to an N-dimensional array.
    ///
    /// For each element, computes the minimum over a local neighborhood.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `size` - Filter window size for each axis
    /// * `mode` - Boundary handling mode
    fn minimum_filter(
        &self,
        input: &Tensor<R>,
        size: &[usize],
        mode: BoundaryMode,
    ) -> Result<Tensor<R>>;

    /// Apply a maximum filter to an N-dimensional array.
    ///
    /// For each element, computes the maximum over a local neighborhood.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `size` - Filter window size for each axis
    /// * `mode` - Boundary handling mode
    fn maximum_filter(
        &self,
        input: &Tensor<R>,
        size: &[usize],
        mode: BoundaryMode,
    ) -> Result<Tensor<R>>;

    /// Apply a percentile filter to an N-dimensional array.
    ///
    /// For each element, computes the given percentile over a local neighborhood.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor
    /// * `percentile` - Percentile to compute (0.0 to 100.0). 0 = minimum, 50 = median, 100 = maximum.
    /// * `size` - Filter window size for each axis
    /// * `mode` - Boundary handling mode
    fn percentile_filter(
        &self,
        input: &Tensor<R>,
        percentile: f64,
        size: &[usize],
        mode: BoundaryMode,
    ) -> Result<Tensor<R>>;
}
