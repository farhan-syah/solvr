//! Robust statistics algorithms.
use crate::DType;

use super::RobustRegressionResult;
use numr::error::Result;
use numr::ops::TensorOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Robust statistics algorithms for tensors.
///
/// These methods are resistant to outliers and violations of distributional assumptions.
pub trait RobustStatisticsAlgorithms<R: Runtime<DType = DType>>: TensorOps<R> {
    /// Trimmed mean.
    ///
    /// Computes the mean after discarding a fraction of values from both tails.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data (1-D tensor)
    /// * `proportiontocut` - Fraction to cut from each tail (must be in [0, 0.5))
    fn trim_mean(&self, x: &Tensor<R>, proportiontocut: f64) -> Result<Tensor<R>>;

    /// Winsorized mean.
    ///
    /// Computes the mean after replacing extreme values with boundary values.
    /// Values below the `proportiontocut` quantile are set to that quantile value,
    /// and values above `1 - proportiontocut` are set to that quantile value.
    ///
    /// # Arguments
    ///
    /// * `x` - Input data (1-D tensor)
    /// * `proportiontocut` - Fraction to winsorize from each tail (must be in [0, 0.5))
    fn winsorized_mean(&self, x: &Tensor<R>, proportiontocut: f64) -> Result<Tensor<R>>;

    /// Median absolute deviation (MAD).
    ///
    /// Computes median(|x - median(x)|). A robust measure of variability.
    /// Optionally scaled by a constant factor (1.4826 for normal consistency).
    ///
    /// # Arguments
    ///
    /// * `x` - Input data (1-D tensor)
    /// * `scale` - If true, multiply by 1.4826 for consistency with standard deviation
    fn median_abs_deviation(&self, x: &Tensor<R>, scale: bool) -> Result<Tensor<R>>;

    /// Siegel repeated medians regression.
    ///
    /// Fits a line y = slope * x + intercept using repeated medians.
    /// More robust than Theil-Sen for datasets with up to ~50% outliers.
    ///
    /// # Arguments
    ///
    /// * `x` - Independent variable (1-D tensor)
    /// * `y` - Dependent variable (1-D tensor, same length as x)
    fn siegelslopes(&self, x: &Tensor<R>, y: &Tensor<R>) -> Result<RobustRegressionResult<R>>;

    /// Theil-Sen slope estimator.
    ///
    /// Fits a line y = slope * x + intercept using the median of pairwise slopes.
    /// Robust to up to ~29% outliers.
    ///
    /// # Arguments
    ///
    /// * `x` - Independent variable (1-D tensor)
    /// * `y` - Dependent variable (1-D tensor, same length as x)
    fn theilslopes(&self, x: &Tensor<R>, y: &Tensor<R>) -> Result<RobustRegressionResult<R>>;
}
