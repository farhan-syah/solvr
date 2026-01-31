//! Distribution traits defining the common interface.

use crate::stats::StatsResult;

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
}

/// Interface for discrete probability distributions.
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
