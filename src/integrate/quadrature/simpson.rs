//! Simpson's rule for numerical integration.
//!
//! Simpson's rule uses parabolic approximation between sample points,
//! achieving O(h⁴) accuracy for smooth functions.

use crate::integrate::error::{IntegrateError, IntegrateResult};

/// Integrate using Simpson's rule.
///
/// Uses the composite Simpson's 1/3 rule for evenly-spaced data.
/// For an odd number of intervals, uses Simpson's 1/3 rule throughout.
/// For an even number of intervals, uses Simpson's 1/3 for most intervals
/// and Simpson's 3/8 rule for the last segment.
///
/// # Arguments
///
/// * `y` - Function values at evenly-spaced sample points
/// * `x` - Sample points (must be same length as `y`, evenly spaced)
///
/// # Returns
///
/// The approximate integral value.
///
/// # Errors
///
/// Returns an error if:
/// - Arrays have different lengths
/// - Arrays have fewer than 3 points
/// - Sample points are not evenly spaced
///
/// # Example
///
/// ```
/// use solvr::integrate::simpson;
///
/// // Integrate y = x^2 from 0 to 1
/// let n = 101;
/// let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
/// let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
///
/// let result = simpson(&y, &x).unwrap();
/// // Exact value is 1/3
/// assert!((result - 1.0/3.0).abs() < 1e-8);
/// ```
pub fn simpson(y: &[f64], x: &[f64]) -> IntegrateResult<f64> {
    let n = y.len();

    if n != x.len() {
        return Err(IntegrateError::InvalidInput {
            context: format!(
                "simpson: x and y must have same length (got {} and {})",
                x.len(),
                n
            ),
        });
    }

    if n < 3 {
        return Err(IntegrateError::InvalidInput {
            context: "simpson: need at least 3 points".to_string(),
        });
    }

    // Check for uniform spacing
    let dx = x[1] - x[0];
    for i in 1..n - 1 {
        let dxi = x[i + 1] - x[i];
        if (dxi - dx).abs() > 1e-10 * dx.abs() {
            return Err(IntegrateError::InvalidInput {
                context: "simpson: sample points must be evenly spaced".to_string(),
            });
        }
    }

    let intervals = n - 1;

    if intervals.is_multiple_of(2) {
        // Even number of intervals: use pure Simpson's 1/3 rule
        simpson_13(y, dx)
    } else {
        // Odd number of intervals
        if n == 3 {
            // n = 3, 2 intervals - cannot use Simpson's 1/3 (needs even intervals)
            // Fall back to trapezoidal rule for this case
            Ok(dx * (y[0] + 4.0 * y[1] + y[2]) / 3.0)
        } else if n == 4 {
            // 3 intervals: use Simpson's 3/8 rule
            simpson_38(y, dx)
        } else {
            // n >= 5: use Simpson's 1/3 for first (n-4) points, 3/8 for last 4
            // This gives us (n-4-1) even intervals for 1/3, then 3 intervals for 3/8
            let first_n = n - 3; // Number of points for first part
            if first_n >= 3 && (first_n - 1).is_multiple_of(2) {
                let first_part = simpson_13(&y[..first_n], dx)?;
                let last_part = simpson_38(&y[first_n - 1..], dx)?;
                Ok(first_part + last_part)
            } else {
                // Fall back: 3/8 for first 4 points, 1/3 for rest
                let first_part = simpson_38(&y[..4], dx)?;
                if n > 4 {
                    let rest_part = simpson_13(&y[3..], dx)?;
                    Ok(first_part + rest_part)
                } else {
                    Ok(first_part)
                }
            }
        }
    }
}

/// Simpson's 1/3 rule for even number of intervals.
fn simpson_13(y: &[f64], dx: f64) -> IntegrateResult<f64> {
    let n = y.len();
    if n < 3 {
        return Err(IntegrateError::InvalidInput {
            context: "simpson_13: need at least 3 points".to_string(),
        });
    }

    // Simpson's 1/3: integral = (dx/3) * (y0 + 4*y1 + 2*y2 + 4*y3 + ... + yn)
    let mut sum = y[0] + y[n - 1];

    // Add 4 * odd terms
    for i in (1..n - 1).step_by(2) {
        sum += 4.0 * y[i];
    }

    // Add 2 * even terms (excluding endpoints)
    for i in (2..n - 1).step_by(2) {
        sum += 2.0 * y[i];
    }

    Ok(dx * sum / 3.0)
}

/// Simpson's 3/8 rule for 4 points.
fn simpson_38(y: &[f64], dx: f64) -> IntegrateResult<f64> {
    if y.len() != 4 {
        return Err(IntegrateError::InvalidInput {
            context: "simpson_38: need exactly 4 points".to_string(),
        });
    }

    // Simpson's 3/8: integral = (3*dx/8) * (y0 + 3*y1 + 3*y2 + y3)
    Ok(3.0 * dx / 8.0 * (y[0] + 3.0 * y[1] + 3.0 * y[2] + y[3]))
}

/// Integrate a function using Simpson's rule.
///
/// This is a convenience function that evaluates the function at evenly-spaced
/// points and applies Simpson's rule.
///
/// # Arguments
///
/// * `f` - Function to integrate
/// * `a` - Lower bound of integration
/// * `b` - Upper bound of integration
/// * `n` - Number of subintervals (must be even)
///
/// # Returns
///
/// The approximate integral value.
///
/// # Example
///
/// ```
/// use solvr::integrate::quadrature::simpson_func;
///
/// // Integrate sin(x) from 0 to pi
/// let result = simpson_func(|x: f64| x.sin(), 0.0, std::f64::consts::PI, 100).unwrap();
/// assert!((result - 2.0).abs() < 1e-6);
/// ```
pub fn simpson_func<F>(f: F, a: f64, b: f64, n: usize) -> IntegrateResult<f64>
where
    F: Fn(f64) -> f64,
{
    if n < 2 {
        return Err(IntegrateError::InvalidParameter {
            parameter: "n".to_string(),
            message: "need at least 2 subintervals".to_string(),
        });
    }

    if a >= b {
        return Err(IntegrateError::InvalidInterval {
            a,
            b,
            context: "simpson_func".to_string(),
        });
    }

    // Use n+1 points for n intervals
    let n_intervals = if n.is_multiple_of(2) { n } else { n + 1 };
    let n_points = n_intervals + 1;
    let dx = (b - a) / n_intervals as f64;

    let y: Vec<f64> = (0..n_points).map(|i| f(a + i as f64 * dx)).collect();

    simpson_13(&y, dx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_simpson_constant() {
        // Integral of constant = constant * width
        let n = 5;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y = vec![3.0; n];

        let result = simpson(&y, &x).unwrap();
        assert!((result - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_simpson_linear() {
        // Integral of y = x from 0 to 4 = 8
        let n = 5;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.clone();

        let result = simpson(&y, &x).unwrap();
        assert!((result - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_simpson_quadratic() {
        // Integral of y = x^2 from 0 to 1 = 1/3 (exact for polynomials up to degree 3)
        let n = 5;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let result = simpson(&y, &x).unwrap();
        assert!((result - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_simpson_cubic() {
        // Integral of y = x^3 from 0 to 1 = 1/4 (exact for polynomials up to degree 3)
        let n = 5;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi * xi).collect();

        let result = simpson(&y, &x).unwrap();
        assert!((result - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_simpson_sin() {
        // Integral of sin(x) from 0 to pi = 2
        // Use 101 points = 100 intervals (even)
        let n = 101;
        let x: Vec<f64> = (0..n).map(|i| PI * i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

        let result = simpson(&y, &x).unwrap();
        // Relax tolerance slightly due to numerical errors
        assert!(
            (result - 2.0).abs() < 1e-6,
            "result = {}, expected 2.0",
            result
        );
    }

    #[test]
    fn test_simpson_func() {
        // Integrate sin(x) from 0 to pi
        let result = simpson_func(|x| x.sin(), 0.0, PI, 100).unwrap();
        // Relax tolerance slightly due to numerical errors
        assert!(
            (result - 2.0).abs() < 1e-6,
            "result = {}, expected 2.0",
            result
        );
    }

    #[test]
    fn test_simpson_exp() {
        // Integral of exp(x) from 0 to 1 = e - 1
        let result = simpson_func(|x| x.exp(), 0.0, 1.0, 100).unwrap();
        let exact = std::f64::consts::E - 1.0;
        assert!((result - exact).abs() < 1e-10);
    }

    #[test]
    fn test_simpson_odd_intervals() {
        // 4 points = 3 intervals (odd)
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let result = simpson(&y, &x).unwrap();
        // Should still give reasonable result
        assert!((result - 9.0).abs() < 0.1); // integral of x^2 from 0 to 3 = 9
    }

    #[test]
    fn test_simpson_errors() {
        // Mismatched lengths
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 2.0];
        assert!(simpson(&y, &x).is_err());

        // Too few points
        let x = vec![0.0, 1.0];
        let y = vec![1.0, 2.0];
        assert!(simpson(&y, &x).is_err());

        // Non-uniform spacing
        let x = vec![0.0, 1.0, 3.0]; // gap of 1, then gap of 2
        let y = vec![1.0, 2.0, 3.0];
        assert!(simpson(&y, &x).is_err());
    }
}
