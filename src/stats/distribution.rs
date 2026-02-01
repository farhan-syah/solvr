//! Distribution traits defining the common interface.
//!
//! All distributions support both scalar and batch (tensor) operations.
//! The tensor methods work on all numr backends (CPU, CUDA, WebGPU).
//!
//! # Performance Note
//!
//! The current tensor methods use scalar evaluation loops as a workaround.
//! For true GPU acceleration, vectorized distribution kernels need to be
//! added to numr (tracked in numr's kernel roadmap). On CPU, performance
//! is acceptable since there's no GPU→CPU→GPU transfer overhead.

use crate::stats::StatsResult;
use numr::error::Result;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Common interface for all probability distributions.
pub trait Distribution {
    /// Get the mean of the distribution.
    fn mean(&self) -> f64;

    /// Get the variance of the distribution.
    fn var(&self) -> f64;

    /// Get the standard deviation of the distribution.
    fn std(&self) -> f64 {
        self.var().sqrt()
    }

    /// Get the Shannon entropy of the distribution (in nats).
    fn entropy(&self) -> f64;

    /// Get the median of the distribution.
    fn median(&self) -> f64;

    /// Get the mode of the distribution.
    fn mode(&self) -> f64;

    /// Get the skewness of the distribution.
    fn skewness(&self) -> f64;

    /// Get the excess kurtosis of the distribution.
    fn kurtosis(&self) -> f64;
}

/// Interface for continuous probability distributions.
///
/// Provides both scalar methods (for single values) and tensor methods
/// (for batch GPU-accelerated computation).
///
/// # Scalar Methods
/// - `pdf(x)` - Probability density at x
/// - `cdf(x)` - Cumulative probability P(X ≤ x)
/// - `ppf(p)` - Quantile function (inverse CDF)
///
/// # Tensor Methods (Batch)
/// - `pdf_tensor(&x, &client)` - PDF for all elements in tensor
/// - `cdf_tensor(&x, &client)` - CDF for all elements in tensor
/// - `ppf_tensor(&p, &client)` - PPF for all elements in tensor
///
/// # Example
///
/// ```ignore
/// use solvr::stats::{Normal, ContinuousDistribution};
/// use numr::runtime::cpu::{CpuClient, CpuDevice};
///
/// let device = CpuDevice::new();
/// let client = CpuClient::new(device.clone());
/// let n = Normal::standard();
///
/// // Scalar
/// let p = n.pdf(0.0);
///
/// // Batch (tensor)
/// let x = Tensor::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
/// let p_batch = n.pdf_tensor(&x, &client).unwrap();
/// ```
pub trait ContinuousDistribution: Distribution {
    /// Probability density function.
    ///
    /// Returns the probability density at point `x`.
    fn pdf(&self, x: f64) -> f64;

    /// Log of the probability density function.
    ///
    /// More numerically stable than `pdf(x).ln()` for extreme values.
    fn log_pdf(&self, x: f64) -> f64 {
        self.pdf(x).ln()
    }

    /// Cumulative distribution function.
    ///
    /// Returns P(X ≤ x).
    fn cdf(&self, x: f64) -> f64;

    /// Survival function.
    ///
    /// Returns P(X > x) = 1 - CDF(x).
    /// More accurate than `1.0 - cdf(x)` for values close to 1.
    fn sf(&self, x: f64) -> f64 {
        1.0 - self.cdf(x)
    }

    /// Log of the cumulative distribution function.
    fn log_cdf(&self, x: f64) -> f64 {
        self.cdf(x).ln()
    }

    /// Percent point function (quantile function / inverse CDF).
    ///
    /// Returns the value x such that P(X ≤ x) = p.
    fn ppf(&self, p: f64) -> StatsResult<f64>;

    /// Inverse survival function.
    ///
    /// Returns the value x such that P(X > x) = p.
    fn isf(&self, p: f64) -> StatsResult<f64> {
        self.ppf(1.0 - p)
    }

    /// Interval containing a given probability mass.
    ///
    /// Returns (a, b) such that P(a ≤ X ≤ b) = alpha, centered around the median.
    fn interval(&self, alpha: f64) -> StatsResult<(f64, f64)> {
        let q = (1.0 - alpha) / 2.0;
        let lower = self.ppf(q)?;
        let upper = self.ppf(1.0 - q)?;
        Ok((lower, upper))
    }

    // ========================================================================
    // Tensor (Batch) Methods
    //
    // NOTE: These methods use scalar evaluation loops as a temporary workaround.
    // For true GPU acceleration, vectorized distribution kernels should be
    // implemented in numr. The pattern below is acceptable because:
    // 1. Distribution evaluation is typically not the bottleneck
    // 2. On CPU there's no transfer overhead
    // 3. The API is correct and will benefit from future numr kernels
    // ========================================================================

    /// Batch probability density function.
    ///
    /// Computes PDF for all elements in the input tensor.
    /// Works on all backends (CPU, CUDA, WebGPU).
    // TODO(numr): Replace with vectorized kernel when available
    fn pdf_tensor<R: Runtime>(
        &self,
        x: &Tensor<R>,
        client: &impl RuntimeClient<R>,
    ) -> Result<Tensor<R>> {
        let x_data: Vec<f64> = x.contiguous().to_vec();
        let result: Vec<f64> = x_data.iter().map(|&xi| self.pdf(xi)).collect();
        Ok(Tensor::<R>::from_slice(&result, x.shape(), client.device()))
    }

    /// Batch log probability density function.
    fn log_pdf_tensor<R: Runtime>(
        &self,
        x: &Tensor<R>,
        client: &impl RuntimeClient<R>,
    ) -> Result<Tensor<R>> {
        let x_data: Vec<f64> = x.contiguous().to_vec();
        let result: Vec<f64> = x_data.iter().map(|&xi| self.log_pdf(xi)).collect();
        Ok(Tensor::<R>::from_slice(&result, x.shape(), client.device()))
    }

    /// Batch cumulative distribution function.
    ///
    /// Computes CDF for all elements in the input tensor.
    fn cdf_tensor<R: Runtime>(
        &self,
        x: &Tensor<R>,
        client: &impl RuntimeClient<R>,
    ) -> Result<Tensor<R>> {
        let x_data: Vec<f64> = x.contiguous().to_vec();
        let result: Vec<f64> = x_data.iter().map(|&xi| self.cdf(xi)).collect();
        Ok(Tensor::<R>::from_slice(&result, x.shape(), client.device()))
    }

    /// Batch survival function.
    fn sf_tensor<R: Runtime>(
        &self,
        x: &Tensor<R>,
        client: &impl RuntimeClient<R>,
    ) -> Result<Tensor<R>> {
        let x_data: Vec<f64> = x.contiguous().to_vec();
        let result: Vec<f64> = x_data.iter().map(|&xi| self.sf(xi)).collect();
        Ok(Tensor::<R>::from_slice(&result, x.shape(), client.device()))
    }

    /// Batch log cumulative distribution function.
    fn log_cdf_tensor<R: Runtime>(
        &self,
        x: &Tensor<R>,
        client: &impl RuntimeClient<R>,
    ) -> Result<Tensor<R>> {
        let x_data: Vec<f64> = x.contiguous().to_vec();
        let result: Vec<f64> = x_data.iter().map(|&xi| self.log_cdf(xi)).collect();
        Ok(Tensor::<R>::from_slice(&result, x.shape(), client.device()))
    }

    /// Batch percent point function (quantile function).
    ///
    /// Computes PPF for all probability values in the input tensor.
    /// Returns error if any probability is outside [0, 1].
    fn ppf_tensor<R: Runtime>(
        &self,
        p: &Tensor<R>,
        client: &impl RuntimeClient<R>,
    ) -> Result<Tensor<R>> {
        let p_data: Vec<f64> = p.contiguous().to_vec();
        let mut result = Vec::with_capacity(p_data.len());
        for &pi in &p_data {
            match self.ppf(pi) {
                Ok(x) => result.push(x),
                Err(e) => {
                    return Err(numr::error::Error::InvalidArgument {
                        arg: "p",
                        reason: format!("ppf_tensor: {}", e),
                    });
                }
            }
        }
        Ok(Tensor::<R>::from_slice(&result, p.shape(), client.device()))
    }

    /// Batch inverse survival function.
    fn isf_tensor<R: Runtime>(
        &self,
        p: &Tensor<R>,
        client: &impl RuntimeClient<R>,
    ) -> Result<Tensor<R>> {
        let p_data: Vec<f64> = p.contiguous().to_vec();
        let mut result = Vec::with_capacity(p_data.len());
        for &pi in &p_data {
            match self.isf(pi) {
                Ok(x) => result.push(x),
                Err(e) => {
                    return Err(numr::error::Error::InvalidArgument {
                        arg: "p",
                        reason: format!("isf_tensor: {}", e),
                    });
                }
            }
        }
        Ok(Tensor::<R>::from_slice(&result, p.shape(), client.device()))
    }
}

/// Interface for discrete probability distributions.
///
/// Provides both scalar methods (for single values) and tensor methods
/// (for batch GPU-accelerated computation).
///
/// # Scalar Methods
/// - `pmf(k)` - Probability mass P(X = k)
/// - `cdf(k)` - Cumulative probability P(X ≤ k)
/// - `ppf(p)` - Quantile function (smallest k with CDF(k) ≥ p)
///
/// # Tensor Methods (Batch)
/// - `pmf_tensor(&k, &client)` - PMF for all elements in tensor
/// - `cdf_tensor(&k, &client)` - CDF for all elements in tensor
/// - `ppf_tensor(&p, &client)` - PPF for all elements in tensor
pub trait DiscreteDistribution: Distribution {
    /// Probability mass function.
    ///
    /// Returns P(X = k).
    fn pmf(&self, k: u64) -> f64;

    /// Log of the probability mass function.
    fn log_pmf(&self, k: u64) -> f64 {
        self.pmf(k).ln()
    }

    /// Cumulative distribution function.
    ///
    /// Returns P(X ≤ k).
    fn cdf(&self, k: u64) -> f64;

    /// Survival function.
    ///
    /// Returns P(X > k) = 1 - CDF(k).
    fn sf(&self, k: u64) -> f64 {
        1.0 - self.cdf(k)
    }

    /// Percent point function (quantile function).
    ///
    /// Returns the smallest k such that P(X ≤ k) ≥ p.
    fn ppf(&self, p: f64) -> StatsResult<u64>;

    /// Inverse survival function.
    fn isf(&self, p: f64) -> StatsResult<u64> {
        self.ppf(1.0 - p)
    }

    // ========================================================================
    // Tensor (Batch) Methods
    // ========================================================================

    /// Batch probability mass function.
    ///
    /// Computes PMF for all elements in the input tensor.
    /// The tensor should contain u64-compatible values (non-negative integers).
    fn pmf_tensor<R: Runtime>(
        &self,
        k: &Tensor<R>,
        client: &impl RuntimeClient<R>,
    ) -> Result<Tensor<R>> {
        let k_data: Vec<f64> = k.contiguous().to_vec();
        let result: Vec<f64> = k_data.iter().map(|&ki| self.pmf(ki as u64)).collect();
        Ok(Tensor::<R>::from_slice(&result, k.shape(), client.device()))
    }

    /// Batch log probability mass function.
    fn log_pmf_tensor<R: Runtime>(
        &self,
        k: &Tensor<R>,
        client: &impl RuntimeClient<R>,
    ) -> Result<Tensor<R>> {
        let k_data: Vec<f64> = k.contiguous().to_vec();
        let result: Vec<f64> = k_data.iter().map(|&ki| self.log_pmf(ki as u64)).collect();
        Ok(Tensor::<R>::from_slice(&result, k.shape(), client.device()))
    }

    /// Batch cumulative distribution function.
    ///
    /// Computes CDF for all elements in the input tensor.
    fn cdf_tensor<R: Runtime>(
        &self,
        k: &Tensor<R>,
        client: &impl RuntimeClient<R>,
    ) -> Result<Tensor<R>> {
        let k_data: Vec<f64> = k.contiguous().to_vec();
        let result: Vec<f64> = k_data.iter().map(|&ki| self.cdf(ki as u64)).collect();
        Ok(Tensor::<R>::from_slice(&result, k.shape(), client.device()))
    }

    /// Batch survival function.
    fn sf_tensor<R: Runtime>(
        &self,
        k: &Tensor<R>,
        client: &impl RuntimeClient<R>,
    ) -> Result<Tensor<R>> {
        let k_data: Vec<f64> = k.contiguous().to_vec();
        let result: Vec<f64> = k_data.iter().map(|&ki| self.sf(ki as u64)).collect();
        Ok(Tensor::<R>::from_slice(&result, k.shape(), client.device()))
    }

    /// Batch percent point function (quantile function).
    ///
    /// Computes PPF for all probability values in the input tensor.
    /// Returns error if any probability is outside [0, 1].
    fn ppf_tensor<R: Runtime>(
        &self,
        p: &Tensor<R>,
        client: &impl RuntimeClient<R>,
    ) -> Result<Tensor<R>> {
        let p_data: Vec<f64> = p.contiguous().to_vec();
        let mut result = Vec::with_capacity(p_data.len());
        for &pi in &p_data {
            match self.ppf(pi) {
                Ok(k) => result.push(k as f64),
                Err(e) => {
                    return Err(numr::error::Error::InvalidArgument {
                        arg: "p",
                        reason: format!("ppf_tensor: {}", e),
                    });
                }
            }
        }
        Ok(Tensor::<R>::from_slice(&result, p.shape(), client.device()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test struct for verifying trait default implementations
    struct MockContinuous;

    impl Distribution for MockContinuous {
        fn mean(&self) -> f64 {
            0.0
        }
        fn var(&self) -> f64 {
            1.0
        }
        fn entropy(&self) -> f64 {
            0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln()
        }
        fn median(&self) -> f64 {
            0.0
        }
        fn mode(&self) -> f64 {
            0.0
        }
        fn skewness(&self) -> f64 {
            0.0
        }
        fn kurtosis(&self) -> f64 {
            0.0
        }
    }

    impl ContinuousDistribution for MockContinuous {
        fn pdf(&self, _x: f64) -> f64 {
            0.5
        }
        fn cdf(&self, x: f64) -> f64 {
            x.clamp(0.0, 1.0)
        }
        fn ppf(&self, p: f64) -> StatsResult<f64> {
            Ok(p)
        }
    }

    #[test]
    fn test_default_implementations() {
        let dist = MockContinuous;
        assert!((dist.std() - 1.0).abs() < 1e-10);
        assert!((dist.sf(0.3) - 0.7).abs() < 1e-10);
        assert!((dist.log_pdf(0.0) - 0.5_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_interval() {
        let dist = MockContinuous;
        let (a, b) = dist.interval(0.9).unwrap();
        assert!((a - 0.05).abs() < 1e-10);
        assert!((b - 0.95).abs() < 1e-10);
    }
}
