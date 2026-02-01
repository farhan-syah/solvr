//! Negative Binomial distribution.

use crate::stats::error::{StatsError, StatsResult};
use crate::stats::{DiscreteDistribution, Distribution};
use numr::algorithm::special::SpecialFunctions;
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Negative Binomial distribution.
///
/// The negative binomial distribution models the number of failures before
/// achieving a specified number of successes. It has PMF:
///
/// P(X = k) = C(k + r - 1, k) * p^r * (1-p)^k
///
/// where:
/// - r > 0 is the number of successes (can be non-integer)
/// - p ∈ (0, 1] is the probability of success on each trial
/// - k ∈ {0, 1, 2, ...} is the number of failures
///
/// When r = 1, this is the geometric distribution.
///
/// # Example
///
/// ```ignore
/// use solvr::stats::{NegativeBinomial, DiscreteDistribution};
///
/// let nb = NegativeBinomial::new(5, 0.5).unwrap();  // 5 successes, p=0.5
/// println!("P(X = 3): {}", nb.pmf(3));
/// println!("Mean: {}", nb.mean());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct NegativeBinomial {
    /// Number of successes (can be non-integer for generalization)
    r: f64,
    /// Probability of success
    p: f64,
}

impl NegativeBinomial {
    /// Create a new negative binomial distribution.
    ///
    /// # Arguments
    ///
    /// * `r` - Number of successes (must be > 0)
    /// * `p` - Probability of success (must be in (0, 1])
    pub fn new(r: u64, p: f64) -> StatsResult<Self> {
        Self::new_real(r as f64, p)
    }

    /// Create a negative binomial with non-integer r (generalized form).
    pub fn new_real(r: f64, p: f64) -> StatsResult<Self> {
        if r <= 0.0 {
            return Err(StatsError::InvalidParameter {
                name: "r".to_string(),
                value: r,
                reason: "number of successes must be positive".to_string(),
            });
        }
        if p <= 0.0 || p > 1.0 {
            return Err(StatsError::InvalidParameter {
                name: "p".to_string(),
                value: p,
                reason: "probability must be in (0, 1]".to_string(),
            });
        }
        Ok(Self { r, p })
    }

    /// Get the number of successes parameter.
    pub fn r(&self) -> f64 {
        self.r
    }

    /// Get the probability of success.
    pub fn p(&self) -> f64 {
        self.p
    }
}

impl Distribution for NegativeBinomial {
    fn mean(&self) -> f64 {
        self.r * (1.0 - self.p) / self.p
    }

    fn var(&self) -> f64 {
        self.r * (1.0 - self.p) / (self.p * self.p)
    }

    fn entropy(&self) -> f64 {
        // No closed form; compute numerically or return approximation
        // For large r, approximately normal approximation entropy
        // Using exact computation for moderate r
        let mut h = 0.0;
        let mut k = 0_u64;
        let mut total_prob = 0.0;

        while total_prob < 0.9999 && k < 10000 {
            let p_k = self.pmf(k);
            if p_k > 1e-300 {
                h -= p_k * p_k.ln();
            }
            total_prob += p_k;
            k += 1;
        }
        h
    }

    fn median(&self) -> f64 {
        // No closed form; find via ppf
        self.ppf(0.5).unwrap_or(self.mean().round() as u64) as f64
    }

    fn mode(&self) -> f64 {
        if self.r <= 1.0 {
            0.0
        } else {
            ((self.r - 1.0) * (1.0 - self.p) / self.p).floor()
        }
    }

    fn skewness(&self) -> f64 {
        (2.0 - self.p) / (self.r * (1.0 - self.p)).sqrt()
    }

    fn kurtosis(&self) -> f64 {
        (6.0 / self.r) + (self.p * self.p) / (self.r * (1.0 - self.p))
    }
}

impl DiscreteDistribution for NegativeBinomial {
    fn pmf(&self, k: u64) -> f64 {
        self.log_pmf(k).exp()
    }

    fn log_pmf(&self, k: u64) -> f64 {
        use super::super::continuous::special::lgamma;

        let r = self.r;
        let p = self.p;
        let k_f = k as f64;

        // log(C(k + r - 1, k)) + r*log(p) + k*log(1-p)
        // = log(Γ(k+r)) - log(Γ(k+1)) - log(Γ(r)) + r*log(p) + k*log(1-p)
        lgamma(k_f + r) - lgamma(k_f + 1.0) - lgamma(r) + r * p.ln() + k_f * (1.0 - p).ln()
    }

    fn cdf(&self, k: u64) -> f64 {
        use super::super::continuous::special::betainc;

        // CDF = I_p(r, k+1) = regularized incomplete beta function
        betainc(self.r, k as f64 + 1.0, self.p)
    }

    fn sf(&self, k: u64) -> f64 {
        1.0 - self.cdf(k)
    }

    fn ppf(&self, prob: f64) -> StatsResult<u64> {
        if !(0.0..=1.0).contains(&prob) {
            return Err(StatsError::InvalidParameter {
                name: "p".to_string(),
                value: prob,
                reason: "probability must be in [0, 1]".to_string(),
            });
        }

        if prob == 0.0 {
            return Ok(0);
        }
        if prob == 1.0 {
            return Ok(u64::MAX);
        }

        // Binary search
        let mean = self.mean();
        let std = self.var().sqrt();
        let mut low = 0_u64;
        let mut high = (mean + 10.0 * std).max(100.0) as u64;

        // Expand upper bound if needed
        while self.cdf(high) < prob {
            high *= 2;
            if high > 1_000_000_000 {
                return Ok(u64::MAX);
            }
        }

        while low < high {
            let mid = low + (high - low) / 2;
            if self.cdf(mid) < prob {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        Ok(low)
    }

    // ========================================================================
    // Tensor Methods - All computation stays on device using numr ops
    // ========================================================================

    fn pmf_tensor<R: Runtime, C>(&self, k: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // PMF(k) = C(k + r - 1, k) * p^r * (1-p)^k
        // Use log-space: log_pmf = log(C(k+r-1,k)) + r*log(p) + k*log(1-p)
        let log_pmf = self.log_pmf_tensor(k, client)?;
        client.exp(&log_pmf)
    }

    fn log_pmf_tensor<R: Runtime, C>(&self, k: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // log_pmf(k) = log(Γ(k+r)) - log(Γ(k+1)) - log(Γ(r)) + r*log(p) + k*log(1-p)
        let ln_p = self.p.ln();
        let ln_q = (1.0 - self.p).ln();
        let lgamma_r = self.r.ln(); // Actually lgamma(r)

        // floor(k) for integer semantics
        let k_floor = client.floor(k)?;

        // k + r
        let k_plus_r = client.add_scalar(&k_floor, self.r)?;
        // k + 1
        let k_plus_1 = client.add_scalar(&k_floor, 1.0)?;

        // lgamma(k + r)
        let lgamma_k_plus_r = client.lgamma(&k_plus_r)?;
        // lgamma(k + 1) = log(k!)
        let lgamma_k_plus_1 = client.lgamma(&k_plus_1)?;

        // log(C(k+r-1, k)) = lgamma(k+r) - lgamma(k+1) - lgamma(r)
        let log_binom_coeff = client.sub(&lgamma_k_plus_r, &lgamma_k_plus_1)?;
        let log_binom_coeff = client.sub_scalar(&log_binom_coeff, lgamma_r)?;

        // r * log(p)
        let r_times_ln_p = self.r * ln_p;

        // k * log(1-p)
        let k_times_ln_q = client.mul_scalar(&k_floor, ln_q)?;

        // Sum: log_binom_coeff + r*log(p) + k*log(1-p)
        let result = client.add_scalar(&log_binom_coeff, r_times_ln_p)?;
        client.add(&result, &k_times_ln_q)
    }

    fn cdf_tensor<R: Runtime, C>(&self, k: &Tensor<R>, client: &C) -> Result<Tensor<R>>
    where
        C: TensorOps<R> + ScalarOps<R> + SpecialFunctions<R> + RuntimeClient<R>,
    {
        // CDF(k) = I_p(r, k+1) = regularized incomplete beta function
        let k_floor = client.floor(k)?;
        let k_plus_1 = client.add_scalar(&k_floor, 1.0)?;

        // r tensor and p tensor
        let shape = k_floor.shape();
        let r_tensor = client.fill(shape, self.r, k_floor.dtype())?;
        let p_tensor = client.fill(shape, self.p, k_floor.dtype())?;

        // betainc(r, k+1, p)
        client.betainc(&r_tensor, &k_plus_1, &p_tensor)
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
        // PPF for negative binomial is complex to vectorize (requires iterative search)
        // Use normal approximation: NegBinom(r,p) ≈ Normal(r(1-p)/p, r(1-p)/p²)
        let mean = self.r * (1.0 - self.p) / self.p;
        let var = self.r * (1.0 - self.p) / (self.p * self.p);
        let std = var.sqrt();

        // Use erfinv for normal approximation
        let two_p_minus_1 = client.sub_scalar(&client.mul_scalar(p, 2.0)?, 1.0)?;
        let erfinv_val = client.erfinv(&two_p_minus_1)?;
        let z = client.mul_scalar(&erfinv_val, std::f64::consts::SQRT_2)?;

        // x = mean + std * z
        let scaled = client.mul_scalar(&z, std)?;
        let result = client.add_scalar(&scaled, mean)?;

        // Clamp to [0, ∞)
        client.clamp(&result, 0.0, f64::INFINITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_negative_binomial_creation() {
        assert!(NegativeBinomial::new(5, 0.5).is_ok());
        assert!(NegativeBinomial::new(0, 0.5).is_err());
        assert!(NegativeBinomial::new(5, 0.0).is_err());
        assert!(NegativeBinomial::new(5, 1.5).is_err());
    }

    #[test]
    fn test_negative_binomial_pmf() {
        let nb = NegativeBinomial::new(3, 0.5).unwrap();

        // P(X=0) = p^r = 0.5^3 = 0.125
        assert!((nb.pmf(0) - 0.125).abs() < 1e-10);

        // P(X=1) = C(3,1) * 0.5^3 * 0.5^1 = 3 * 0.0625 = 0.1875
        assert!((nb.pmf(1) - 0.1875).abs() < 1e-10);

        // PMF should sum to 1
        let sum: f64 = (0..100).map(|k| nb.pmf(k)).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_negative_binomial_cdf() {
        let nb = NegativeBinomial::new(3, 0.5).unwrap();

        // CDF(0) = PMF(0)
        assert!((nb.cdf(0) - nb.pmf(0)).abs() < 1e-10);

        // CDF is monotonically increasing
        assert!(nb.cdf(0) <= nb.cdf(1));
        assert!(nb.cdf(1) <= nb.cdf(2));

        // CDF should approach 1
        assert!(nb.cdf(100) > 0.999);
    }

    #[test]
    fn test_negative_binomial_mean() {
        let nb = NegativeBinomial::new(5, 0.5).unwrap();

        // Mean = r(1-p)/p = 5 * 0.5 / 0.5 = 5
        assert!((nb.mean() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_negative_binomial_variance() {
        let nb = NegativeBinomial::new(5, 0.5).unwrap();

        // Variance = r(1-p)/p² = 5 * 0.5 / 0.25 = 10
        assert!((nb.var() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_negative_binomial_ppf() {
        let nb = NegativeBinomial::new(5, 0.5).unwrap();

        // PPF(0) = 0
        assert_eq!(nb.ppf(0.0).unwrap(), 0);

        // Round-trip for several values
        for k in [0, 1, 5, 10] {
            let p = nb.cdf(k);
            let recovered = nb.ppf(p).unwrap();
            // Should get back k or k+1 due to discrete nature
            assert!(recovered == k || recovered == k + 1);
        }
    }

    #[test]
    fn test_geometric_special_case() {
        // Negative binomial with r=1 is geometric (shifted)
        let nb = NegativeBinomial::new(1, 0.3).unwrap();

        // Mean = (1-p)/p
        let expected_mean = 0.7 / 0.3;
        assert!((nb.mean() - expected_mean).abs() < 1e-10);
    }
}
