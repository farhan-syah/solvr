//! Wishart distribution.

use super::special;
use crate::stats::distribution::Distribution;
use crate::stats::error::{StatsError, StatsResult};

/// Wishart distribution.
///
/// The Wishart distribution is a matrix-variate distribution over p×p
/// positive-definite matrices. It is parameterized by degrees of freedom ν
/// and a p×p positive-definite scale matrix V.
///
/// PDF: f(X) = |X|^((ν-p-1)/2) exp(-tr(V⁻¹X)/2) / (2^(νp/2) |V|^(ν/2) Γₚ(ν/2))
///
/// where Γₚ is the multivariate gamma function.
#[derive(Debug, Clone)]
pub struct Wishart {
    /// Degrees of freedom (ν)
    df: f64,
    /// Scale matrix (p×p, flattened row-major)
    scale: Vec<f64>,
    /// Dimension
    p: usize,
    /// Log-determinant of scale matrix
    log_det_scale: f64,
    /// Log normalizing constant
    log_norm: f64,
}

impl Wishart {
    /// Create a new Wishart distribution.
    ///
    /// # Arguments
    ///
    /// * `df` - Degrees of freedom (must be >= p)
    /// * `scale` - p×p positive-definite scale matrix (flattened row-major)
    /// * `p` - Matrix dimension
    pub fn new(df: f64, scale: Vec<f64>, p: usize) -> StatsResult<Self> {
        if p == 0 {
            return Err(StatsError::InvalidParameter {
                name: "p".to_string(),
                value: 0.0,
                reason: "dimension must be positive".to_string(),
            });
        }
        if scale.len() != p * p {
            return Err(StatsError::InvalidParameter {
                name: "scale".to_string(),
                value: scale.len() as f64,
                reason: format!("scale matrix must have p*p = {} elements", p * p),
            });
        }
        if df < p as f64 {
            return Err(StatsError::InvalidParameter {
                name: "df".to_string(),
                value: df,
                reason: format!("degrees of freedom must be >= p = {}", p),
            });
        }

        let log_det_scale = log_det(&scale, p);
        if log_det_scale.is_nan() || log_det_scale.is_infinite() {
            return Err(StatsError::InvalidParameter {
                name: "scale".to_string(),
                value: log_det_scale,
                reason: "scale matrix must be positive definite".to_string(),
            });
        }

        // Log normalizing constant:
        // log(2^(νp/2)) + (ν/2)*log|V| + log(Γₚ(ν/2))
        let half_df = df / 2.0;
        let pf = p as f64;
        let log_norm = half_df * pf * std::f64::consts::LN_2
            + half_df * log_det_scale
            + log_multivariate_gamma(half_df, p);

        Ok(Self {
            df,
            scale,
            p,
            log_det_scale,
            log_norm,
        })
    }

    /// Get degrees of freedom.
    pub fn df(&self) -> f64 {
        self.df
    }

    /// Get the scale matrix (flattened).
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }

    /// Get the dimension.
    pub fn p(&self) -> usize {
        self.p
    }

    /// Mean matrix: E[X] = ν * V
    pub fn mean_matrix(&self) -> Vec<f64> {
        self.scale.iter().map(|&v| self.df * v).collect()
    }

    /// Mode matrix: (ν - p - 1) * V for ν >= p + 1
    pub fn mode_matrix(&self) -> Option<Vec<f64>> {
        let pf = self.p as f64;
        if self.df < pf + 1.0 {
            return None;
        }
        let factor = self.df - pf - 1.0;
        Some(self.scale.iter().map(|&v| factor * v).collect())
    }

    /// Log-PDF of a p×p positive-definite matrix X (flattened row-major).
    pub fn log_pdf(&self, x: &[f64]) -> f64 {
        assert_eq!(x.len(), self.p * self.p, "x must be p×p matrix");

        let log_det_x = log_det(x, self.p);
        if log_det_x.is_nan() || log_det_x.is_infinite() {
            return f64::NEG_INFINITY;
        }

        let pf = self.p as f64;

        // (ν - p - 1)/2 * log|X|
        let term1 = (self.df - pf - 1.0) / 2.0 * log_det_x;

        // -tr(V⁻¹ X) / 2
        let scale_inv = matrix_inverse(&self.scale, self.p);
        let trace = matrix_trace_product(&scale_inv, x, self.p);
        let term2 = -trace / 2.0;

        term1 + term2 - self.log_norm
    }

    /// PDF of a p×p positive-definite matrix X.
    pub fn pdf(&self, x: &[f64]) -> f64 {
        self.log_pdf(x).exp()
    }
}

impl Distribution for Wishart {
    fn mean(&self) -> f64 {
        // Trace of mean matrix / p = ν * tr(V) / p
        let trace: f64 = (0..self.p).map(|i| self.scale[i * self.p + i]).sum();
        self.df * trace / self.p as f64
    }

    fn var(&self) -> f64 {
        // Var(X_ii) = 2ν * V_ii²
        let v_00 = self.scale[0];
        2.0 * self.df * v_00 * v_00
    }

    fn entropy(&self) -> f64 {
        let pf = self.p as f64;
        let half_df = self.df / 2.0;

        self.log_norm + (pf + 1.0 - self.df) / 2.0 * self.log_det_scale + half_df * pf
            - (self.df - pf - 1.0) / 2.0
                * (0..self.p)
                    .map(|i| special::digamma((self.df - i as f64) / 2.0))
                    .sum::<f64>()
    }

    fn median(&self) -> f64 {
        // No closed form; approximate as mean
        self.mean()
    }

    fn mode(&self) -> f64 {
        let pf = self.p as f64;
        if self.df >= pf + 1.0 {
            let trace: f64 = (0..self.p).map(|i| self.scale[i * self.p + i]).sum();
            (self.df - pf - 1.0) * trace / pf
        } else {
            0.0
        }
    }

    fn skewness(&self) -> f64 {
        // Approximate: skewness of diagonal element
        (8.0 / self.df).sqrt()
    }

    fn kurtosis(&self) -> f64 {
        // Excess kurtosis of diagonal element: 12/ν
        12.0 / self.df
    }
}

// --- Matrix helpers ---

/// Log-determinant via LU decomposition with partial pivoting.
fn log_det(m: &[f64], n: usize) -> f64 {
    let mut a = m.to_vec();
    let mut sign = 1.0_f64;

    for k in 0..n {
        // Find pivot
        let mut max_val = a[k * n + k].abs();
        let mut max_row = k;
        for i in (k + 1)..n {
            let v = a[i * n + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }

        if max_val < 1e-15 {
            return f64::NEG_INFINITY; // Singular
        }

        if max_row != k {
            for j in 0..n {
                a.swap(k * n + j, max_row * n + j);
            }
            sign = -sign;
        }

        let pivot = a[k * n + k];
        for i in (k + 1)..n {
            let factor = a[i * n + k] / pivot;
            for j in (k + 1)..n {
                a[i * n + j] -= factor * a[k * n + j];
            }
        }
    }

    let mut log_det = if sign > 0.0 { 0.0 } else { return f64::NAN };
    for i in 0..n {
        let d = a[i * n + i];
        if d <= 0.0 {
            return f64::NEG_INFINITY;
        }
        log_det += d.ln();
    }
    log_det
}

/// Matrix inverse via Gauss-Jordan elimination.
fn matrix_inverse(m: &[f64], n: usize) -> Vec<f64> {
    let mut aug = vec![0.0; n * 2 * n];

    // Build augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = m[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    let w = 2 * n;
    for k in 0..n {
        // Find pivot
        let mut max_row = k;
        let mut max_val = aug[k * w + k].abs();
        for i in (k + 1)..n {
            let v = aug[i * w + k].abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }

        if max_row != k {
            for j in 0..w {
                aug.swap(k * w + j, max_row * w + j);
            }
        }

        let pivot = aug[k * w + k];
        for j in 0..w {
            aug[k * w + j] /= pivot;
        }

        for i in 0..n {
            if i != k {
                let factor = aug[i * w + k];
                for j in 0..w {
                    aug[i * w + j] -= factor * aug[k * w + j];
                }
            }
        }
    }

    // Extract inverse
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * w + n + j];
        }
    }
    inv
}

/// Trace of product A * B.
fn matrix_trace_product(a: &[f64], b: &[f64], n: usize) -> f64 {
    let mut trace = 0.0;
    for i in 0..n {
        for k in 0..n {
            trace += a[i * n + k] * b[k * n + i];
        }
    }
    trace
}

/// Log of the multivariate gamma function.
///
/// Γₚ(a) = π^(p(p-1)/4) ∏_{j=1}^{p} Γ(a + (1-j)/2)
fn log_multivariate_gamma(a: f64, p: usize) -> f64 {
    let pf = p as f64;
    let mut result = pf * (pf - 1.0) / 4.0 * std::f64::consts::PI.ln();
    for j in 1..=p {
        result += special::lgamma(a + (1.0 - j as f64) / 2.0);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(p: usize) -> Vec<f64> {
        let mut m = vec![0.0; p * p];
        for i in 0..p {
            m[i * p + i] = 1.0;
        }
        m
    }

    #[test]
    fn test_wishart_creation() {
        let w = Wishart::new(3.0, identity(2), 2).unwrap();
        assert_eq!(w.p(), 2);
        assert!((w.df() - 3.0).abs() < 1e-10);

        // df < p should fail
        assert!(Wishart::new(1.0, identity(2), 2).is_err());
        // Wrong matrix size
        assert!(Wishart::new(3.0, vec![1.0], 2).is_err());
    }

    #[test]
    fn test_wishart_mean() {
        let w = Wishart::new(5.0, identity(2), 2).unwrap();
        let mean = w.mean_matrix();
        // Mean = ν * I = 5 * I
        assert!((mean[0] - 5.0).abs() < 1e-10); // (0,0)
        assert!((mean[1] - 0.0).abs() < 1e-10); // (0,1)
        assert!((mean[3] - 5.0).abs() < 1e-10); // (1,1)
    }

    #[test]
    fn test_wishart_mode() {
        let w = Wishart::new(5.0, identity(2), 2).unwrap();
        let mode = w.mode_matrix().unwrap();
        // Mode = (ν - p - 1) * V = 2 * I
        assert!((mode[0] - 2.0).abs() < 1e-10);
        assert!((mode[3] - 2.0).abs() < 1e-10);

        // df = p, no mode
        let w2 = Wishart::new(2.0, identity(2), 2).unwrap();
        assert!(w2.mode_matrix().is_none());
    }

    #[test]
    fn test_wishart_pdf_positive() {
        let w = Wishart::new(5.0, identity(2), 2).unwrap();
        // Test at identity
        let pdf_val = w.pdf(&identity(2));
        assert!(pdf_val > 0.0);
        assert!(pdf_val.is_finite());
    }

    #[test]
    fn test_wishart_log_det() {
        // Identity matrix: det = 1, log_det = 0
        assert!((log_det(&identity(3), 3) - 0.0).abs() < 1e-10);

        // 2x2 diagonal matrix [[2,0],[0,3]]: det = 6
        let m = vec![2.0, 0.0, 0.0, 3.0];
        assert!((log_det(&m, 2) - 6.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_wishart_matrix_inverse() {
        let m = vec![2.0, 1.0, 1.0, 3.0];
        let inv = matrix_inverse(&m, 2);
        // For [[2,1],[1,3]], det=5, inv = [[3/5, -1/5],[-1/5, 2/5]]
        assert!((inv[0] - 0.6).abs() < 1e-10);
        assert!((inv[1] - (-0.2)).abs() < 1e-10);
        assert!((inv[2] - (-0.2)).abs() < 1e-10);
        assert!((inv[3] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_wishart_distribution_trait() {
        let w = Wishart::new(10.0, identity(3), 3).unwrap();
        assert!(w.mean().is_finite());
        assert!(w.var() > 0.0);
        assert!(w.entropy().is_finite());
        assert!(w.skewness() > 0.0);
        assert!(w.kurtosis() > 0.0);
    }
}
