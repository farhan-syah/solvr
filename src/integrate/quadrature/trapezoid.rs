//! Trapezoidal rule for numerical integration.
//!
//! The trapezoidal rule approximates the integral by summing trapezoid areas.
//! It has O(h²) accuracy for smooth functions.

use crate::integrate::error::{IntegrateError, IntegrateResult};

/// Integrate using the composite trapezoidal rule.
///
/// Computes the definite integral of `y` values over `x` using the trapezoidal rule.
/// This is suitable for sampled data where you have discrete (x, y) pairs.
///
/// # Arguments
///
/// * `y` - Function values at sample points
/// * `x` - Sample points (must be same length as `y`)
///
/// # Returns
///
/// The approximate integral value.
///
/// # Errors
///
/// Returns an error if:
/// - Arrays have different lengths
/// - Arrays have fewer than 2 points
///
/// # Example
///
/// ```
/// use solvr::integrate::trapezoid;
///
/// // Integrate y = x^2 from 0 to 1 using 101 points
/// let n = 101;
/// let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
/// let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
///
/// let result = trapezoid(&y, &x).unwrap();
/// // Exact value is 1/3 ≈ 0.333...
/// assert!((result - 1.0/3.0).abs() < 0.001);
/// ```
pub fn trapezoid(y: &[f64], x: &[f64]) -> IntegrateResult<f64> {
    let n = y.len();

    if n != x.len() {
        return Err(IntegrateError::InvalidInput {
            context: format!(
                "trapezoid: x and y must have same length (got {} and {})",
                x.len(),
                n
            ),
        });
    }

    if n < 2 {
        return Err(IntegrateError::InvalidInput {
            context: "trapezoid: need at least 2 points".to_string(),
        });
    }

    let mut integral = 0.0;
    for i in 0..n - 1 {
        let dx = x[i + 1] - x[i];
        integral += 0.5 * dx * (y[i] + y[i + 1]);
    }

    Ok(integral)
}

/// Integrate using the trapezoidal rule with uniform spacing.
///
/// A convenience function when sample points are uniformly spaced.
///
/// # Arguments
///
/// * `y` - Function values at sample points
/// * `dx` - Spacing between consecutive sample points
///
/// # Returns
///
/// The approximate integral value.
///
/// # Errors
///
/// Returns an error if `y` has fewer than 2 points.
///
/// # Example
///
/// ```
/// use solvr::integrate::quadrature::trapezoid_uniform;
///
/// // Integrate y = x^2 from 0 to 1 with dx = 0.01
/// let n = 101;
/// let dx = 1.0 / (n - 1) as f64;
/// let y: Vec<f64> = (0..n).map(|i| {
///     let x = i as f64 * dx;
///     x * x
/// }).collect();
///
/// let result = trapezoid_uniform(&y, dx).unwrap();
/// assert!((result - 1.0/3.0).abs() < 0.001);
/// ```
pub fn trapezoid_uniform(y: &[f64], dx: f64) -> IntegrateResult<f64> {
    let n = y.len();

    if n < 2 {
        return Err(IntegrateError::InvalidInput {
            context: "trapezoid_uniform: need at least 2 points".to_string(),
        });
    }

    // For uniform spacing: integral = dx * (y[0]/2 + y[1] + ... + y[n-2] + y[n-1]/2)
    let mut sum = 0.5 * (y[0] + y[n - 1]);
    for yi in y.iter().take(n - 1).skip(1) {
        sum += *yi;
    }

    Ok(dx * sum)
}

/// Compute the cumulative integral using the trapezoidal rule.
///
/// Returns the cumulative integral at each point, starting from 0.
///
/// # Arguments
///
/// * `y` - Function values at sample points
/// * `x` - Sample points (must be same length as `y`)
///
/// # Returns
///
/// A vector where element `i` is the integral from `x[0]` to `x[i]`.
///
/// # Errors
///
/// Returns an error if arrays have different lengths or fewer than 2 points.
///
/// # Example
///
/// ```
/// use solvr::integrate::cumulative_trapezoid;
///
/// // Cumulative integral of y = 2x (antiderivative is x^2)
/// let x = vec![0.0, 1.0, 2.0, 3.0];
/// let y = vec![0.0, 2.0, 4.0, 6.0];
///
/// let result = cumulative_trapezoid(&y, &x).unwrap();
/// // result[i] should be x[i]^2
/// assert!((result[0] - 0.0).abs() < 1e-10);
/// assert!((result[1] - 1.0).abs() < 1e-10);
/// assert!((result[2] - 4.0).abs() < 1e-10);
/// assert!((result[3] - 9.0).abs() < 1e-10);
/// ```
pub fn cumulative_trapezoid(y: &[f64], x: &[f64]) -> IntegrateResult<Vec<f64>> {
    let n = y.len();

    if n != x.len() {
        return Err(IntegrateError::InvalidInput {
            context: format!(
                "cumulative_trapezoid: x and y must have same length (got {} and {})",
                x.len(),
                n
            ),
        });
    }

    if n < 2 {
        return Err(IntegrateError::InvalidInput {
            context: "cumulative_trapezoid: need at least 2 points".to_string(),
        });
    }

    let mut result = Vec::with_capacity(n);
    result.push(0.0);

    let mut cumsum = 0.0;
    for i in 0..n - 1 {
        let dx = x[i + 1] - x[i];
        cumsum += 0.5 * dx * (y[i] + y[i + 1]);
        result.push(cumsum);
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_trapezoid_constant() {
        // Integral of constant function = constant * width
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![5.0, 5.0, 5.0, 5.0, 5.0];

        let result = trapezoid(&y, &x).unwrap();
        assert!((result - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_trapezoid_linear() {
        // Integral of y = x from 0 to 1 = 0.5 (exact for linear functions)
        let n = 11;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = x.clone();

        let result = trapezoid(&y, &x).unwrap();
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_trapezoid_quadratic() {
        // Integral of y = x^2 from 0 to 1 = 1/3
        let n = 1001;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

        let result = trapezoid(&y, &x).unwrap();
        assert!((result - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_trapezoid_sin() {
        // Integral of sin(x) from 0 to pi = 2
        let n = 1001;
        let x: Vec<f64> = (0..n).map(|i| PI * i as f64 / (n - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

        let result = trapezoid(&y, &x).unwrap();
        assert!((result - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_trapezoid_uniform() {
        let n = 1001;
        let dx = 1.0 / (n - 1) as f64;
        let y: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 * dx;
                x * x
            })
            .collect();

        let result = trapezoid_uniform(&y, dx).unwrap();
        assert!((result - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_cumulative_trapezoid() {
        // Integral of y = 2x is x^2
        let x = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi).collect();

        let result = cumulative_trapezoid(&y, &x).unwrap();

        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[1] - 0.25).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10);
        assert!((result[3] - 2.25).abs() < 1e-10);
        assert!((result[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trapezoid_errors() {
        // Mismatched lengths
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![1.0, 2.0];
        assert!(trapezoid(&y, &x).is_err());

        // Too few points
        let x = vec![0.0];
        let y = vec![1.0];
        assert!(trapezoid(&y, &x).is_err());
    }
}
