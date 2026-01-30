//! Adaptive quadrature methods.
//!
//! These methods automatically subdivide the integration interval to achieve
//! a specified error tolerance.

use crate::integrate::error::{IntegrateError, IntegrateResult};

/// Options for adaptive quadrature.
#[derive(Debug, Clone)]
pub struct QuadOptions {
    /// Relative tolerance (default: 1e-8)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-8)
    pub atol: f64,
    /// Maximum number of subdivisions (default: 50)
    pub limit: usize,
}

impl Default for QuadOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-8,
            limit: 50,
        }
    }
}

/// Result of adaptive quadrature.
#[derive(Debug, Clone)]
pub struct QuadResult {
    /// Computed integral value
    pub integral: f64,
    /// Estimated absolute error
    pub error: f64,
    /// Number of function evaluations
    pub neval: usize,
    /// Whether integration converged
    pub converged: bool,
}

/// Adaptive Gauss-Kronrod quadrature.
///
/// Uses the G7-K15 rule (7-point Gauss, 15-point Kronrod) with adaptive
/// interval subdivision to achieve the requested tolerance.
///
/// # Arguments
///
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `options` - Quadrature options
///
/// # Returns
///
/// A [`QuadResult`] containing the integral, error estimate, and diagnostics.
///
/// # Example
///
/// ```
/// use solvr::integrate::{quad, QuadOptions};
///
/// // Integrate sin(x) from 0 to pi
/// let result = quad(|x| x.sin(), 0.0, std::f64::consts::PI, &QuadOptions::default()).unwrap();
/// assert!((result.integral - 2.0).abs() < 1e-10);
/// assert!(result.converged);
/// ```
pub fn quad<F>(f: F, a: f64, b: f64, options: &QuadOptions) -> IntegrateResult<QuadResult>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrateError::InvalidInterval {
            a,
            b,
            context: "quad".to_string(),
        });
    }

    if options.limit == 0 {
        return Err(IntegrateError::InvalidParameter {
            parameter: "limit".to_string(),
            message: "must be at least 1".to_string(),
        });
    }

    // Use a work queue for intervals that need refinement
    let mut intervals: Vec<(f64, f64, f64, f64)> = Vec::new(); // (a, b, integral, error)
    let mut neval = 0;

    // Initial evaluation
    let (integral, error, evals) = gauss_kronrod_15(&f, a, b);
    neval += evals;

    intervals.push((a, b, integral, error));

    let mut total_integral = integral;
    let mut total_error = error;
    let mut subdivisions = 0;

    while subdivisions < options.limit {
        // Check convergence
        let tolerance = options.atol + options.rtol * total_integral.abs();
        if total_error <= tolerance {
            return Ok(QuadResult {
                integral: total_integral,
                error: total_error,
                neval,
                converged: true,
            });
        }

        // Find interval with largest error
        let max_idx = intervals
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.3.partial_cmp(&b.1.3).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let (ia, ib, old_integral, old_error) = intervals.swap_remove(max_idx);

        // Subdivide at midpoint
        let mid = (ia + ib) / 2.0;

        let (int1, err1, evals1) = gauss_kronrod_15(&f, ia, mid);
        let (int2, err2, evals2) = gauss_kronrod_15(&f, mid, ib);
        neval += evals1 + evals2;

        // Update totals
        total_integral = total_integral - old_integral + int1 + int2;
        total_error = total_error - old_error + err1 + err2;

        intervals.push((ia, mid, int1, err1));
        intervals.push((mid, ib, int2, err2));

        subdivisions += 1;
    }

    // Didn't converge within limit
    Ok(QuadResult {
        integral: total_integral,
        error: total_error,
        neval,
        converged: false,
    })
}

/// Gauss-Kronrod 15-point rule (G7-K15).
///
/// Returns (integral, error, neval).
fn gauss_kronrod_15<F>(f: &F, a: f64, b: f64) -> (f64, f64, usize)
where
    F: Fn(f64) -> f64,
{
    // Kronrod nodes (15 points, including 7 Gauss points)
    const XGK: [f64; 15] = [
        -0.9914553711208126,
        -0.9491079123427585,
        -0.8648644233597691,
        -0.7415311855993944,
        -0.5860872354676911,
        -0.4058451513773972,
        -0.2077849550078985,
        0.0,
        0.2077849550078985,
        0.4058451513773972,
        0.5860872354676911,
        0.7415311855993944,
        0.8648644233597691,
        0.9491079123427585,
        0.9914553711208126,
    ];

    // Kronrod weights (15 points)
    const WGK: [f64; 15] = [
        0.022935322010529224,
        0.063_092_092_629_978_56,
        0.10479001032225018,
        0.14065325971552592,
        0.169_004_726_639_267_9,
        0.190_350_578_064_785_4,
        0.20443294007529889,
        0.20948214108472782,
        0.20443294007529889,
        0.190_350_578_064_785_4,
        0.169_004_726_639_267_9,
        0.14065325971552592,
        0.10479001032225018,
        0.063_092_092_629_978_56,
        0.022935322010529224,
    ];

    // Gauss weights (7 points at indices 1,3,5,7,9,11,13)
    const WG: [f64; 7] = [
        0.129_484_966_168_869_7,
        0.27970539148927664,
        0.381_830_050_505_118_9,
        0.417_959_183_673_469_4,
        0.381_830_050_505_118_9,
        0.27970539148927664,
        0.129_484_966_168_869_7,
    ];

    let mid = (a + b) / 2.0;
    let half_width = (b - a) / 2.0;

    // Evaluate function at all 15 points
    let mut fvals = [0.0; 15];
    for (i, &x) in XGK.iter().enumerate() {
        fvals[i] = f(mid + half_width * x);
    }

    // Compute Kronrod (15-point) result
    let mut result_kronrod = 0.0;
    for (i, &fval) in fvals.iter().enumerate() {
        result_kronrod += WGK[i] * fval;
    }
    result_kronrod *= half_width;

    // Compute Gauss (7-point) result using odd-indexed points
    let mut result_gauss = 0.0;
    for (i, &w) in WG.iter().enumerate() {
        result_gauss += w * fvals[2 * i + 1];
    }
    result_gauss *= half_width;

    // Error estimate
    let error = (result_kronrod - result_gauss).abs();

    (result_kronrod, error, 15)
}

/// Options for Romberg integration.
#[derive(Debug, Clone)]
pub struct RombergOptions {
    /// Relative tolerance (default: 1e-8)
    pub rtol: f64,
    /// Absolute tolerance (default: 1e-8)
    pub atol: f64,
    /// Maximum number of extrapolation levels (default: 20)
    pub max_levels: usize,
}

impl Default for RombergOptions {
    fn default() -> Self {
        Self {
            rtol: 1e-8,
            atol: 1e-8,
            max_levels: 20,
        }
    }
}

/// Romberg integration using Richardson extrapolation.
///
/// Applies Richardson extrapolation to the trapezoidal rule to achieve
/// high accuracy for smooth functions.
///
/// # Arguments
///
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `options` - Integration options
///
/// # Returns
///
/// A [`QuadResult`] containing the integral, error estimate, and diagnostics.
///
/// # Example
///
/// ```
/// use solvr::integrate::{romberg, RombergOptions};
///
/// // Integrate exp(x) from 0 to 1 = e - 1
/// let result = romberg(|x| x.exp(), 0.0, 1.0, &RombergOptions::default()).unwrap();
/// let exact = std::f64::consts::E - 1.0;
/// assert!((result.integral - exact).abs() < 1e-10);
/// ```
pub fn romberg<F>(f: F, a: f64, b: f64, options: &RombergOptions) -> IntegrateResult<QuadResult>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrateError::InvalidInterval {
            a,
            b,
            context: "romberg".to_string(),
        });
    }

    if options.max_levels == 0 {
        return Err(IntegrateError::InvalidParameter {
            parameter: "max_levels".to_string(),
            message: "must be at least 1".to_string(),
        });
    }

    let mut neval = 0;
    let h = b - a;

    // Romberg table (only need current and previous row)
    let mut r_prev = vec![0.0; options.max_levels];
    let mut r_curr = vec![0.0; options.max_levels];

    // Initial trapezoidal estimate (1 interval)
    let fa = f(a);
    let fb = f(b);
    neval += 2;
    r_prev[0] = h * (fa + fb) / 2.0;

    // Limit k to prevent overflow in 1 << k and 4^j calculations
    let max_k = options.max_levels.min(30); // 2^30 is about 1 billion intervals

    for k in 1..max_k {
        // Compute trapezoidal estimate with 2^k intervals
        let n: u64 = 1 << k; // 2^k = number of intervals
        let h_k = h / n as f64; // step size for this level

        // Add new midpoints: we need to evaluate at x = a + (2i+1) * h_k
        // These are the points that weren't evaluated in the previous level
        let num_new_points = 1u64 << (k - 1); // 2^(k-1) new points
        let mut sum = 0.0;
        for i in 0..num_new_points {
            let x = a + (2 * i + 1) as f64 * h_k;
            sum += f(x);
        }
        neval += num_new_points as usize;

        // New trapezoidal estimate: T_k = T_{k-1}/2 + h_k * sum
        r_curr[0] = r_prev[0] / 2.0 + h_k * sum;

        // Richardson extrapolation
        for j in 1..=k {
            let factor = 4.0_f64.powi(j as i32); // 4^j
            r_curr[j] = (factor * r_curr[j - 1] - r_prev[j - 1]) / (factor - 1.0);
        }

        // Check convergence - compare with previous level at same column
        let error = (r_curr[k] - r_prev[k - 1]).abs();
        let tolerance = options.atol + options.rtol * r_curr[k].abs();

        if error <= tolerance {
            return Ok(QuadResult {
                integral: r_curr[k],
                error,
                neval,
                converged: true,
            });
        }

        std::mem::swap(&mut r_prev, &mut r_curr);
    }

    // Return best estimate even if not converged
    let k = max_k - 1;
    let error = if k > 0 {
        (r_prev[k] - r_prev[k - 1]).abs()
    } else {
        r_prev[0].abs() * 0.1 // Rough estimate
    };

    Ok(QuadResult {
        integral: r_prev[k],
        error,
        neval,
        converged: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_quad_polynomial() {
        // Integrate x^4 from 0 to 1 = 0.2
        let result = quad(|x| x.powi(4), 0.0, 1.0, &QuadOptions::default()).unwrap();
        assert!((result.integral - 0.2).abs() < 1e-10);
        assert!(result.converged);
    }

    #[test]
    fn test_quad_trig() {
        // Integrate sin(x) from 0 to pi = 2
        let result = quad(|x| x.sin(), 0.0, PI, &QuadOptions::default()).unwrap();
        assert!((result.integral - 2.0).abs() < 1e-10);
        assert!(result.converged);
    }

    #[test]
    fn test_quad_exp() {
        // Integrate exp(x) from 0 to 1 = e - 1
        let result = quad(|x| x.exp(), 0.0, 1.0, &QuadOptions::default()).unwrap();
        let exact = std::f64::consts::E - 1.0;
        assert!((result.integral - exact).abs() < 1e-10);
        assert!(result.converged);
    }

    #[test]
    fn test_quad_oscillatory() {
        // Integrate sin(10x) from 0 to pi (harder case)
        let result = quad(|x| (10.0 * x).sin(), 0.0, PI, &QuadOptions::default()).unwrap();
        // Exact: (1 - cos(10*pi)) / 10 = 0
        assert!(result.integral.abs() < 1e-8);
        assert!(result.converged);
    }

    #[test]
    fn test_quad_with_peak() {
        // Integrate 1/(1 + 100*(x-0.5)^2) from 0 to 1
        // This has a sharp peak at x = 0.5
        let options = QuadOptions {
            limit: 100,
            ..Default::default()
        };
        let result = quad(
            |x| 1.0 / (1.0 + 100.0 * (x - 0.5).powi(2)),
            0.0,
            1.0,
            &options,
        )
        .unwrap();
        // Exact: arctan(5)/5 + arctan(5)/5 = 2*arctan(5)/5 ≈ 0.2952
        // Actually: integral = (arctan(10*(1-0.5)) - arctan(10*(0-0.5)))/10
        //                    = (arctan(5) - arctan(-5))/10 = 2*arctan(5)/10 = arctan(5)/5
        let exact = 2.0 * 5.0_f64.atan() / 10.0;
        assert!(
            (result.integral - exact).abs() < 1e-6,
            "got {}, expected {}",
            result.integral,
            exact
        );
    }

    #[test]
    fn test_romberg_polynomial() {
        // Romberg is very accurate for polynomials
        let result = romberg(|x| x.powi(4), 0.0, 1.0, &RombergOptions::default()).unwrap();
        assert!(
            (result.integral - 0.2).abs() < 1e-8,
            "integral = {}, expected 0.2",
            result.integral
        );
        assert!(result.converged);
    }

    #[test]
    fn test_romberg_exp() {
        let result = romberg(|x| x.exp(), 0.0, 1.0, &RombergOptions::default()).unwrap();
        let exact = std::f64::consts::E - 1.0;
        assert!(
            (result.integral - exact).abs() < 1e-8,
            "integral = {}, expected {}",
            result.integral,
            exact
        );
        assert!(result.converged);
    }

    #[test]
    fn test_romberg_trig() {
        let result = romberg(|x| x.sin(), 0.0, PI, &RombergOptions::default()).unwrap();
        assert!(
            (result.integral - 2.0).abs() < 1e-8,
            "integral = {}, expected 2.0",
            result.integral
        );
        assert!(result.converged);
    }

    #[test]
    fn test_quad_error_handling() {
        // Invalid interval
        let result = quad(|x| x, 2.0, 1.0, &QuadOptions::default());
        assert!(result.is_err());

        // Zero limit
        let options = QuadOptions {
            limit: 0,
            ..Default::default()
        };
        let result = quad(|x| x, 0.0, 1.0, &options);
        assert!(result.is_err());
    }
}
