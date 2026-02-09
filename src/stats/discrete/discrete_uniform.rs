//! Discrete Uniform distribution.

use crate::stats::error::{StatsError, StatsResult};
use crate::stats::{DiscreteDistribution, Distribution};
use numr::algorithm::special::SpecialFunctions;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Discrete Uniform distribution.
///
/// The discrete uniform distribution assigns equal probability to each
/// integer value in a specified range [low, high]. It has PMF:
///
/// P(X = k) = 1 / (high - low + 1)  for k ∈ {low, low+1, ..., high}
///
/// # Example
///
/// ```
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use solvr::stats::{DiscreteUniform, DiscreteDistribution, Distribution};
///
/// // Fair die
/// let die = DiscreteUniform::new(1, 6)?;
/// println!("P(X = 3): {}", die.pmf(3));  // 1/6
/// println!("Mean: {}", die.mean());      // 3.5
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone, Copy)]
pub struct DiscreteUniform {
    /// Lower bound (inclusive)
    low: i64,
    /// Upper bound (inclusive)
    high: i64,
    /// Number of possible values
    n: u64,
}

impl DiscreteUniform {
    /// Create a new discrete uniform distribution.
    ///
    /// # Arguments
    ///
    /// * `low` - Lower bound (inclusive)
    /// * `high` - Upper bound (inclusive, must be >= low)
    pub fn new(low: i64, high: i64) -> StatsResult<Self> {
        if high < low {
            return Err(StatsError::InvalidParameter {
                name: "high".to_string(),
                value: high as f64,
                reason: format!("upper bound must be >= lower bound ({})", low),
            });
        }
        let n = (high - low + 1) as u64;
        Ok(Self { low, high, n })
    }

    /// Create a discrete uniform over [0, n-1] (like randint).
    pub fn randint(n: u64) -> StatsResult<Self> {
        if n == 0 {
            return Err(StatsError::InvalidParameter {
                name: "n".to_string(),
                value: 0.0,
                reason: "n must be positive".to_string(),
            });
        }
        Self::new(0, (n - 1) as i64)
    }

    /// Get the lower bound.
    pub fn low(&self) -> i64 {
        self.low
    }

    /// Get the upper bound.
    pub fn high(&self) -> i64 {
        self.high
    }

    /// Get the number of possible values.
    pub fn n(&self) -> u64 {
        self.n
    }

    /// Check if a value is in the support.
    fn in_support(&self, k: i64) -> bool {
        k >= self.low && k <= self.high
    }
}

impl Distribution for DiscreteUniform {
    fn mean(&self) -> f64 {
        (self.low + self.high) as f64 / 2.0
    }

    fn var(&self) -> f64 {
        // Variance = (n² - 1) / 12 where n = high - low + 1
        let n = self.n as f64;
        (n * n - 1.0) / 12.0
    }

    fn entropy(&self) -> f64 {
        // Entropy = ln(n)
        (self.n as f64).ln()
    }

    fn median(&self) -> f64 {
        (self.low + self.high) as f64 / 2.0
    }

    fn mode(&self) -> f64 {
        // All values are modes; return the mean
        self.mean()
    }

    fn skewness(&self) -> f64 {
        // Symmetric distribution
        0.0
    }

    fn kurtosis(&self) -> f64 {
        // Excess kurtosis = -6(n² + 1) / (5(n² - 1))
        let n = self.n as f64;
        let n2 = n * n;
        -6.0 * (n2 + 1.0) / (5.0 * (n2 - 1.0))
    }
}

impl DiscreteDistribution for DiscreteUniform {
    fn pmf(&self, k: u64) -> f64 {
        let k_signed = k as i64;
        if self.in_support(k_signed) {
            1.0 / self.n as f64
        } else {
            0.0
        }
    }

    fn log_pmf(&self, k: u64) -> f64 {
        let k_signed = k as i64;
        if self.in_support(k_signed) {
            -(self.n as f64).ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    fn cdf(&self, k: u64) -> f64 {
        let k_signed = k as i64;
        if k_signed < self.low {
            0.0
        } else if k_signed >= self.high {
            1.0
        } else {
            (k_signed - self.low + 1) as f64 / self.n as f64
        }
    }

    fn sf(&self, k: u64) -> f64 {
        1.0 - self.cdf(k)
    }

    fn ppf(&self, p: f64) -> StatsResult<u64> {
        if !(0.0..=1.0).contains(&p) {
            return Err(StatsError::InvalidParameter {
                name: "p".to_string(),
                value: p,
                reason: "probability must be in [0, 1]".to_string(),
            });
        }

        if p == 0.0 {
            return Ok(self.low as u64);
        }
        if p == 1.0 {
            return Ok(self.high as u64);
        }

        // k such that CDF(k) >= p
        // CDF(k) = (k - low + 1) / n >= p
        // k >= low + n*p - 1
        let k = self.low + (self.n as f64 * p).ceil() as i64 - 1;
        let k = k.max(self.low).min(self.high);
        Ok(k as u64)
    }

    // ========================================================================
    // Tensor Methods - All computation stays on device using numr ops
    // ========================================================================

    fn pmf_tensor<R: Runtime, C>(&self, k: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // PMF is constant: 1 / (high - low + 1) for k in [low, high], else 0
        // For all values, return 1/n
        // Note: For proper handling of out-of-support values, the user should
        // ensure k is in the valid range, or post-process with masking
        let pmf_const = 1.0 / (self.n as f64);
        let shape = k.shape();
        client.fill(shape, pmf_const, k.dtype())
    }

    fn log_pmf_tensor<R: Runtime, C>(&self, k: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // log(PMF) = -ln(n)
        // Note: For out-of-support values, handling requires conditional logic
        let log_pmf_const = -(self.n as f64).ln();
        let shape = k.shape();
        client.fill(shape, log_pmf_const, k.dtype())
    }

    fn cdf_tensor<R: Runtime, C>(&self, k: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // CDF(k) = (floor(k) - low + 1) / n, clamped to [0, 1]
        let low_f = self.low as f64;
        let n_f = self.n as f64;

        // floor(k)
        let k_floor = client.floor(k)?;

        // (k_floor - low + 1) / n
        let adjusted = client.sub_scalar(&k_floor, low_f - 1.0)?;
        let cdf_val = client.div_scalar(&adjusted, n_f)?;

        // Clamp to [0, 1]
        client.clamp(&cdf_val, 0.0, 1.0)
    }

    fn sf_tensor<R: Runtime, C>(&self, k: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // SF(k) = 1 - CDF(k)
        let cdf = self.cdf_tensor(k, client)?;
        client.sub_scalar(&client.mul_scalar(&cdf, -1.0)?, -1.0)
    }

    fn ppf_tensor<R: Runtime, C>(&self, p: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // PPF(p) = low + ceil(n*p) - 1
        let low_f = self.low as f64;
        let high_f = self.high as f64;
        let n_f = self.n as f64;

        // n * p
        let n_times_p = client.mul_scalar(p, n_f)?;
        // ceil(n*p) - 1 + low
        let ceiled = client.ceil(&n_times_p)?;
        let ppf_val = client.add_scalar(&ceiled, low_f - 1.0)?;

        // Clamp to [low, high]
        client.clamp(&ppf_val, low_f, high_f)
    }
}

/// Convenience methods for creating common distributions.
impl DiscreteUniform {
    /// Create a fair coin (0 or 1).
    pub fn coin() -> Self {
        Self::new(0, 1).unwrap()
    }

    /// Create a fair die with n sides (1 to n).
    pub fn die(n: u64) -> StatsResult<Self> {
        if n == 0 {
            return Err(StatsError::InvalidParameter {
                name: "n".to_string(),
                value: 0.0,
                reason: "number of sides must be positive".to_string(),
            });
        }
        Self::new(1, n as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_uniform_creation() {
        assert!(DiscreteUniform::new(1, 6).is_ok());
        assert!(DiscreteUniform::new(5, 5).is_ok()); // Single value
        assert!(DiscreteUniform::new(6, 1).is_err()); // Invalid range
    }

    #[test]
    fn test_discrete_uniform_pmf() {
        let d = DiscreteUniform::new(1, 6).unwrap();

        // All values in range have equal probability
        for k in 1..=6 {
            assert!((d.pmf(k) - 1.0 / 6.0).abs() < 1e-10);
        }

        // Values outside range have zero probability
        assert!((d.pmf(0) - 0.0).abs() < 1e-10);
        assert!((d.pmf(7) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_discrete_uniform_cdf() {
        let d = DiscreteUniform::new(1, 6).unwrap();

        // CDF should be monotonically increasing
        assert!((d.cdf(1) - 1.0 / 6.0).abs() < 1e-10);
        assert!((d.cdf(2) - 2.0 / 6.0).abs() < 1e-10);
        assert!((d.cdf(3) - 3.0 / 6.0).abs() < 1e-10);
        assert!((d.cdf(6) - 1.0).abs() < 1e-10);

        // CDF outside range
        assert!((d.cdf(0) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_discrete_uniform_mean() {
        let d = DiscreteUniform::new(1, 6).unwrap();
        assert!((d.mean() - 3.5).abs() < 1e-10);

        let d = DiscreteUniform::new(0, 10).unwrap();
        assert!((d.mean() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_discrete_uniform_variance() {
        let d = DiscreteUniform::new(1, 6).unwrap();
        // Var = (6² - 1) / 12 = 35/12 ≈ 2.917
        assert!((d.var() - 35.0 / 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_discrete_uniform_ppf() {
        let d = DiscreteUniform::new(1, 6).unwrap();

        assert_eq!(d.ppf(0.0).unwrap(), 1);
        assert_eq!(d.ppf(1.0).unwrap(), 6);

        // Round-trip
        for k in 1..=6_u64 {
            let p = d.cdf(k);
            let recovered = d.ppf(p).unwrap();
            assert!(recovered == k || recovered == k + 1);
        }
    }

    #[test]
    fn test_discrete_uniform_entropy() {
        let d = DiscreteUniform::new(1, 6).unwrap();
        // Entropy = ln(6)
        assert!((d.entropy() - 6.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_discrete_uniform_skewness() {
        let d = DiscreteUniform::new(1, 6).unwrap();
        // Symmetric distribution has zero skewness
        assert!((d.skewness() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_discrete_uniform_die() {
        let d6 = DiscreteUniform::die(6).unwrap();
        assert_eq!(d6.low(), 1);
        assert_eq!(d6.high(), 6);
        assert_eq!(d6.n(), 6);
    }

    #[test]
    fn test_discrete_uniform_coin() {
        let coin = DiscreteUniform::coin();
        assert_eq!(coin.low(), 0);
        assert_eq!(coin.high(), 1);
        assert!((coin.pmf(0) - 0.5).abs() < 1e-10);
        assert!((coin.pmf(1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_discrete_uniform_randint() {
        let d = DiscreteUniform::randint(10).unwrap();
        assert_eq!(d.low(), 0);
        assert_eq!(d.high(), 9);
        assert_eq!(d.n(), 10);
    }

    #[test]
    fn test_discrete_uniform_single_value() {
        let d = DiscreteUniform::new(5, 5).unwrap();
        assert!((d.pmf(5) - 1.0).abs() < 1e-10);
        assert!((d.mean() - 5.0).abs() < 1e-10);
        assert!((d.var() - 0.0).abs() < 1e-10);
    }
}
