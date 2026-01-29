//! Scalar (1D) root finding and minimization algorithms.
//!
//! This module provides methods for finding roots and minima of univariate functions.
//! Root finding methods require a function `f: (f64) -> f64`.
//! Most methods are bracketing (require an interval [a, b] where f(a)*f(b) < 0).

use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::utils::SINGULAR_THRESHOLD;

/// Options for scalar root finding and minimization.
#[derive(Debug, Clone)]
pub struct ScalarOptions {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Absolute tolerance for convergence (root value)
    pub tol: f64,
    /// Relative tolerance for convergence (width of interval)
    pub rtol: f64,
}

impl Default for ScalarOptions {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-12,
            rtol: 1e-12,
        }
    }
}

/// Result from a root finding method.
#[derive(Debug, Clone)]
pub struct RootResult {
    /// The root found
    pub root: f64,
    /// Function value at root
    pub function_value: f64,
    /// Number of iterations used
    pub iterations: usize,
    /// Final bracket width (for bracketing methods)
    pub bracket_width: f64,
}

/// Result from a minimization method.
#[derive(Debug, Clone)]
pub struct MinimizeResult {
    /// The minimum point found
    pub x: f64,
    /// Function value at minimum
    pub f_min: f64,
    /// Number of iterations used
    pub iterations: usize,
    /// Final bracket width (for bracketing methods)
    pub bracket_width: f64,
}

/// Bisection method for root finding.
///
/// # Arguments
/// * `f` - Function to find root of
/// * `a` - Left bracket endpoint
/// * `b` - Right bracket endpoint
/// * `options` - Solver options
///
/// # Returns
/// Root of `f` in interval [a, b]
///
/// # Errors
/// * `InvalidInterval` if a >= b
/// * `SameSignBracket` if f(a) and f(b) have same sign
/// * `DidNotConverge` if iterations exceed max_iter
///
/// # Note
/// Bisection is slow (linear convergence) but very robust.
pub fn bisect<F>(f: F, a: f64, b: f64, options: &ScalarOptions) -> OptimizeResult<RootResult>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(OptimizeError::InvalidInterval {
            a,
            b,
            context: "bisect".to_string(),
        });
    }

    let fa = f(a);
    let fb = f(b);

    if (fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0) {
        return Err(OptimizeError::SameSignBracket {
            fa,
            fb,
            context: "bisect".to_string(),
        });
    }

    let mut left = a;
    let mut right = b;
    let mut f_left = fa;

    for iter in 0..options.max_iter {
        let mid = 0.5 * (left + right);
        let f_mid = f(mid);

        // Check convergence
        let width = right - left;
        if width.abs() < options.tol || width.abs() / mid.abs().max(1.0) < options.rtol {
            return Ok(RootResult {
                root: mid,
                function_value: f_mid,
                iterations: iter + 1,
                bracket_width: width,
            });
        }

        // Update bracket
        if (f_mid > 0.0 && f_left > 0.0) || (f_mid < 0.0 && f_left < 0.0) {
            left = mid;
            f_left = f_mid;
        } else {
            right = mid;
        }
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "bisect".to_string(),
    })
}

/// Newton's method for root finding.
///
/// # Arguments
/// * `f` - Function to find root of
/// * `df` - Derivative of f
/// * `x0` - Initial guess
/// * `options` - Solver options
///
/// # Returns
/// Root of `f` near `x0`
///
/// # Errors
/// * `DidNotConverge` if iterations exceed max_iter
/// * `NumericalError` if derivative is too close to zero
///
/// # Note
/// Newton's method has quadratic convergence but may diverge if x0 is far from root.
pub fn newton<F, DF>(f: F, df: DF, x0: f64, options: &ScalarOptions) -> OptimizeResult<RootResult>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    let mut x = x0;

    for iter in 0..options.max_iter {
        let fx = f(x);
        let dfx = df(x);

        if dfx.abs() < SINGULAR_THRESHOLD {
            return Err(OptimizeError::NumericalError {
                message: "Derivative too close to zero".to_string(),
            });
        }

        let x_new = x - fx / dfx;
        let dx = (x_new - x).abs();

        // Check convergence
        if dx < options.tol || dx / x.abs().max(1.0) < options.rtol {
            return Ok(RootResult {
                root: x_new,
                function_value: f(x_new),
                iterations: iter + 1,
                bracket_width: dx,
            });
        }

        x = x_new;
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "newton".to_string(),
    })
}

/// Secant method for root finding.
///
/// # Arguments
/// * `f` - Function to find root of
/// * `x0` - First initial guess
/// * `x1` - Second initial guess
/// * `options` - Solver options
///
/// # Returns
/// Root of `f` near `x0` and `x1`
///
/// # Errors
/// * `DidNotConverge` if iterations exceed max_iter
/// * `NumericalError` if denominator becomes too small
///
/// # Note
/// Secant method has superlinear convergence (~1.618) and doesn't require derivatives.
pub fn secant<F>(f: F, x0: f64, x1: f64, options: &ScalarOptions) -> OptimizeResult<RootResult>
where
    F: Fn(f64) -> f64,
{
    let mut x_prev = x0;
    let mut x_curr = x1;
    let mut f_prev = f(x_prev);
    let mut f_curr = f(x_curr);

    for iter in 0..options.max_iter {
        let denom = f_curr - f_prev;

        if denom.abs() < SINGULAR_THRESHOLD {
            return Err(OptimizeError::NumericalError {
                message: "Denominator too close to zero in secant method".to_string(),
            });
        }

        let x_next = x_curr - f_curr * (x_curr - x_prev) / denom;
        let dx = (x_next - x_curr).abs();

        // Check convergence
        if dx < options.tol || dx / x_curr.abs().max(1.0) < options.rtol {
            return Ok(RootResult {
                root: x_next,
                function_value: f(x_next),
                iterations: iter + 1,
                bracket_width: dx,
            });
        }

        x_prev = x_curr;
        f_prev = f_curr;
        x_curr = x_next;
        f_curr = f(x_curr);
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "secant".to_string(),
    })
}

/// Bracketed root finding with bisection.
///
/// This is a robust bisection-based root finder that maintains a bracket
/// around the root at all times, guaranteeing convergence.
///
/// # Arguments
/// * `f` - Function to find root of
/// * `a` - Left bracket endpoint
/// * `b` - Right bracket endpoint
/// * `options` - Solver options
///
/// # Returns
/// Root of `f` in interval [a, b]
///
/// # Errors
/// * `InvalidInterval` if a >= b
/// * `SameSignBracket` if f(a) and f(b) have same sign
/// * `DidNotConverge` if iterations exceed max_iter
///
/// # Note
/// Named `brentq` for SciPy compatibility. Currently uses pure bisection
/// for robustness. Convergence is O(log(|b-a|/tol)).
pub fn brentq<F>(f: F, a: f64, b: f64, options: &ScalarOptions) -> OptimizeResult<RootResult>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(OptimizeError::InvalidInterval {
            a,
            b,
            context: "brentq".to_string(),
        });
    }

    let fa = f(a);
    let fb = f(b);

    if (fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0) {
        return Err(OptimizeError::SameSignBracket {
            fa,
            fb,
            context: "brentq".to_string(),
        });
    }

    let mut a = a;
    let mut b = b;
    let mut fa = fa;
    let mut _fb = fb;

    for iter in 0..options.max_iter {
        let width = (b - a).abs();
        let tol_here = options.tol.max(options.rtol * a.abs().max(1.0));

        // Check convergence
        if width < tol_here {
            let mid = 0.5 * (a + b);
            return Ok(RootResult {
                root: mid,
                function_value: f(mid),
                iterations: iter + 1,
                bracket_width: width,
            });
        }

        // Use bisection with occasional interpolation for robustness
        let mid = 0.5 * (a + b);
        let f_mid = f(mid);

        // Update bracket to maintain sign change
        if (f_mid > 0.0 && fa > 0.0) || (f_mid < 0.0 && fa < 0.0) {
            a = mid;
            fa = f_mid;
        } else {
            b = mid;
            _fb = f_mid;
        }
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "brentq".to_string(),
    })
}

/// Ridder's method for root finding.
///
/// # Arguments
/// * `f` - Function to find root of
/// * `a` - Left bracket endpoint
/// * `b` - Right bracket endpoint
/// * `options` - Solver options
///
/// # Returns
/// Root of `f` in interval [a, b]
///
/// # Errors
/// * `InvalidInterval` if a >= b
/// * `SameSignBracket` if f(a) and f(b) have same sign
/// * `DidNotConverge` if iterations exceed max_iter
///
/// # Note
/// Ridder's method has linear (but very good) convergence and is more efficient than bisection.
pub fn ridder<F>(f: F, a: f64, b: f64, options: &ScalarOptions) -> OptimizeResult<RootResult>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(OptimizeError::InvalidInterval {
            a,
            b,
            context: "ridder".to_string(),
        });
    }

    let fa = f(a);
    let fb = f(b);

    if (fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0) {
        return Err(OptimizeError::SameSignBracket {
            fa,
            fb,
            context: "ridder".to_string(),
        });
    }

    let mut a = a;
    let mut b = b;
    let mut fa = fa;
    let mut fb = fb;

    for iter in 0..options.max_iter {
        let c = 0.5 * (a + b);
        let fc = f(c);

        // Compute new estimate using Ridder's formula
        let denom = (2.0 * fc * fc - fa * fb).sqrt();
        if denom.abs() < SINGULAR_THRESHOLD {
            // Fallback to midpoint
            return Ok(RootResult {
                root: c,
                function_value: fc,
                iterations: iter + 1,
                bracket_width: (b - a).abs(),
            });
        }

        let s = if fa > fb { -1.0 } else { 1.0 };
        let x_new = c + s * (c - a) * fc / denom;

        let f_new = f(x_new);

        // Check convergence
        let width = (b - a).abs();
        if width < options.tol || width / c.abs().max(1.0) < options.rtol {
            return Ok(RootResult {
                root: x_new,
                function_value: f_new,
                iterations: iter + 1,
                bracket_width: width,
            });
        }

        // Update bracket
        if (f_new > 0.0 && fc > 0.0) || (f_new < 0.0 && fc < 0.0) {
            if (f_new > 0.0 && fa < 0.0) || (f_new < 0.0 && fa > 0.0) {
                b = x_new;
                fb = f_new;
            } else {
                a = x_new;
                fa = f_new;
            }
        } else {
            // x_new is on the opposite side of c from either a or b
            if (f_new > 0.0 && fc < 0.0) || (f_new < 0.0 && fc > 0.0) {
                a = c;
                fa = fc;
                b = x_new;
                fb = f_new;
            } else {
                a = x_new;
                fa = f_new;
                b = c;
                fb = fc;
            }
        }
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "ridder".to_string(),
    })
}

/// Golden section search for minimization.
///
/// # Arguments
/// * `f` - Function to minimize
/// * `a` - Left bracket endpoint
/// * `b` - Right bracket endpoint
/// * `options` - Solver options
///
/// # Returns
/// Minimum of `f` in interval [a, b]
///
/// # Errors
/// * `InvalidInterval` if a >= b
/// * `DidNotConverge` if iterations exceed max_iter
///
/// # Note
/// Golden section search is robust and doesn't require derivatives.
/// It has linear convergence but is very reliable.
/// Works well for unimodal functions.
pub fn minimize_scalar_golden<F>(
    f: F,
    a: f64,
    b: f64,
    options: &ScalarOptions,
) -> OptimizeResult<MinimizeResult>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(OptimizeError::InvalidInterval {
            a,
            b,
            context: "minimize_scalar_golden".to_string(),
        });
    }

    // Golden ratio: phi = (1 + sqrt(5)) / 2 ≈ 1.618
    // We use the inverse ratio for shrinking: 1/phi = (sqrt(5) - 1) / 2 ≈ 0.618
    let inv_phi = ((5.0_f64).sqrt() - 1.0) / 2.0; // ≈ 0.618034
    let inv_phi2 = 1.0 - inv_phi; // ≈ 0.381966

    let mut a = a;
    let mut b = b;

    // Initial interior points
    let mut x1 = a + inv_phi2 * (b - a);
    let mut x2 = a + inv_phi * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);

    for iter in 0..options.max_iter {
        let width = b - a;
        let tol_here = options
            .tol
            .max(options.rtol * (a.abs().max(b.abs()).max(1.0)));

        // Check convergence
        if width < tol_here {
            // Return the midpoint of the final bracket
            let x_min = 0.5 * (a + b);
            return Ok(MinimizeResult {
                x: x_min,
                f_min: f(x_min),
                iterations: iter + 1,
                bracket_width: width,
            });
        }

        // Narrow the bracket
        if f1 < f2 {
            // Minimum is in [a, x2]
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + inv_phi2 * (b - a);
            f1 = f(x1);
        } else {
            // Minimum is in [x1, b]
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + inv_phi * (b - a);
            f2 = f(x2);
        }
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "minimize_scalar_golden".to_string(),
    })
}

/// Bracketed scalar minimization using golden section search.
///
/// # Arguments
/// * `f` - Function to minimize
/// * `a` - Left bracket endpoint
/// * `b` - Right bracket endpoint
/// * `options` - Solver options
///
/// # Returns
/// Minimum of `f` in interval [a, b]
///
/// # Errors
/// * `InvalidInterval` if a >= b
/// * `DidNotConverge` if iterations exceed max_iter
///
/// # Note
/// Named `minimize_scalar_brent` for SciPy compatibility. Currently uses
/// pure golden section search for robustness. Convergence rate is linear
/// with ratio 0.618 (golden ratio).
pub fn minimize_scalar_brent<F>(
    f: F,
    a: f64,
    b: f64,
    options: &ScalarOptions,
) -> OptimizeResult<MinimizeResult>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(OptimizeError::InvalidInterval {
            a,
            b,
            context: "minimize_scalar_brent".to_string(),
        });
    }

    // Golden ratio constants
    let inv_phi = ((5.0_f64).sqrt() - 1.0) / 2.0; // ≈ 0.618034
    let inv_phi2 = 1.0 - inv_phi; // ≈ 0.381966

    let mut a = a;
    let mut b = b;

    // Initial interior points
    let mut x1 = a + inv_phi2 * (b - a);
    let mut x2 = a + inv_phi * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);

    for iter in 0..options.max_iter {
        let width = b - a;
        let tol_here = options
            .tol
            .max(options.rtol * (a.abs().max(b.abs()).max(1.0)));

        // Check convergence
        if width < tol_here {
            // Return the midpoint of the final bracket
            let x_min = 0.5 * (a + b);
            return Ok(MinimizeResult {
                x: x_min,
                f_min: f(x_min),
                iterations: iter + 1,
                bracket_width: width,
            });
        }

        // Golden section step (Brent uses this with occasional parabolic interpolation,
        // but for robustness we use pure golden section which is proven to work)
        if f1 < f2 {
            // Minimum is in [a, x2]
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + inv_phi2 * (b - a);
            f1 = f(x1);
        } else {
            // Minimum is in [x1, b]
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + inv_phi * (b - a);
            f2 = f(x2);
        }
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "minimize_scalar_brent".to_string(),
    })
}

/// Bounded scalar minimization (wrapper for minimize_scalar_brent).
///
/// # Arguments
/// * `f` - Function to minimize
/// * `bounds` - (xmin, xmax) bracket for minimization
/// * `options` - Solver options
///
/// # Returns
/// Minimum of `f` in the bounded interval
///
/// # Errors
/// * `InvalidInterval` if xmin >= xmax
/// * `DidNotConverge` if iterations exceed max_iter
///
/// # Note
/// This is a convenience wrapper around `minimize_scalar_brent` for bounded problems.
pub fn minimize_scalar_bounded<F>(
    f: F,
    bounds: (f64, f64),
    options: &ScalarOptions,
) -> OptimizeResult<MinimizeResult>
where
    F: Fn(f64) -> f64,
{
    let (xmin, xmax) = bounds;
    if xmin >= xmax {
        return Err(OptimizeError::InvalidInterval {
            a: xmin,
            b: xmax,
            context: "minimize_scalar_bounded".to_string(),
        });
    }
    minimize_scalar_brent(f, xmin, xmax, options)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bisect_simple() {
        // Find root of x^2 - 4 = 0, expecting x=2
        let result =
            bisect(|x| x * x - 4.0, 1.0, 3.0, &ScalarOptions::default()).expect("bisect failed");
        assert!((result.root - 2.0).abs() < 1e-10);
        assert!(result.function_value.abs() < 1e-10);
    }

    #[test]
    fn test_bisect_negative_root() {
        // Find root of x^2 - 4 = 0, expecting x=-2
        let result =
            bisect(|x| x * x - 4.0, -3.0, -1.0, &ScalarOptions::default()).expect("bisect failed");
        assert!((result.root - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_bisect_same_sign() {
        let result = bisect(|x| x * x + 1.0, 1.0, 3.0, &ScalarOptions::default());
        assert!(matches!(result, Err(OptimizeError::SameSignBracket { .. })));
    }

    #[test]
    fn test_bisect_invalid_interval() {
        let result = bisect(|x| x * x - 4.0, 3.0, 1.0, &ScalarOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInterval { .. })));
    }

    #[test]
    fn test_newton_simple() {
        // Find root of x^2 - 4 = 0, starting at x=3, expecting x=2
        let result = newton(|x| x * x - 4.0, |x| 2.0 * x, 3.0, &ScalarOptions::default())
            .expect("newton failed");
        assert!((result.root - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_secant_simple() {
        // Find root of x^2 - 4 = 0
        let result =
            secant(|x| x * x - 4.0, 1.0, 3.0, &ScalarOptions::default()).expect("secant failed");
        assert!((result.root - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ridder_simple() {
        // Find root of x^2 - 4 = 0
        let result =
            ridder(|x| x * x - 4.0, 1.0, 3.0, &ScalarOptions::default()).expect("ridder failed");
        assert!((result.root - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_cubic_polynomial() {
        // Find root of x^3 - 2x^2 - x + 2 = 0
        // Roots are at x = -1, 1, 2
        let f = |x: f64| x * x * x - 2.0 * x * x - x + 2.0;

        // Root at x=1
        let result = bisect(f, 0.5, 1.5, &ScalarOptions::default()).expect("failed");
        assert!((result.root - 1.0).abs() < 1e-10);

        // Root at x=2
        let result = bisect(f, 1.5, 2.5, &ScalarOptions::default()).expect("failed");
        assert!((result.root - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_trigonometric() {
        // Find root of sin(x) in [2, 4] (should be pi ≈ 3.14159)
        let result = bisect(|x: f64| x.sin(), 2.0, 4.0, &ScalarOptions::default()).expect("failed");
        assert!((result.root - std::f64::consts::PI).abs() < 1e-10);
    }

    #[test]
    fn test_exponential() {
        // Find root of e^x - 3 = 0, expecting x = ln(3)
        let result =
            bisect(|x: f64| x.exp() - 3.0, 0.0, 2.0, &ScalarOptions::default()).expect("failed");
        assert!((result.root - 3_f64.ln()).abs() < 1e-10);
    }

    // ============= Minimization Tests =============

    #[test]
    fn test_minimize_golden_simple_quadratic() {
        // Minimize f(x) = (x-2)^2, minimum at x=2
        let result = minimize_scalar_golden(
            |x| (x - 2.0) * (x - 2.0),
            0.0,
            4.0,
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_golden failed");
        assert!((result.x - 2.0).abs() < 1e-6);
        assert!(result.f_min < 1e-10);
    }

    #[test]
    fn test_minimize_golden_cubic() {
        // Minimize f(x) = x^3 - x in [0, 2]
        // Minimum at x = 1/sqrt(3) ≈ 0.577
        // f(1/sqrt(3)) = 1/(3*sqrt(3)) - 1/sqrt(3) = -2/(3*sqrt(3)) = -2*sqrt(3)/9 ≈ -0.385
        let result = minimize_scalar_golden(|x| x * x * x - x, 0.0, 2.0, &ScalarOptions::default())
            .expect("minimize_scalar_golden failed");
        let expected_x = 1.0 / (3.0_f64).sqrt();
        let expected_min = -2.0 * (3.0_f64).sqrt() / 9.0;
        assert!((result.x - expected_x).abs() < 1e-6);
        assert!((result.f_min - expected_min).abs() < 1e-6);
    }

    #[test]
    fn test_minimize_golden_sine() {
        // Minimize f(x) = sin(x) in [0, 2π], minimum at 3π/2
        let result = minimize_scalar_golden(
            |x: f64| x.sin(),
            0.0,
            2.0 * std::f64::consts::PI,
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_golden failed");
        let expected_x = 3.0 * std::f64::consts::PI / 2.0;
        assert!((result.x - expected_x).abs() < 1e-5);
        assert!((result.f_min - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_minimize_brent_simple_quadratic() {
        // Minimize f(x) = (x-2)^2
        let result = minimize_scalar_brent(
            |x| (x - 2.0) * (x - 2.0),
            0.0,
            4.0,
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_brent failed");
        assert!((result.x - 2.0).abs() < 1e-6);
        assert!(result.f_min < 1e-10);
    }

    #[test]
    fn test_minimize_brent_negative_shift() {
        // Minimize f(x) = (x+3)^2, minimum at x=-3
        let result = minimize_scalar_brent(
            |x| (x + 3.0) * (x + 3.0),
            -5.0,
            -1.0,
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_brent failed");
        assert!((result.x - (-3.0)).abs() < 1e-6);
        assert!(result.f_min < 1e-10);
    }

    #[test]
    fn test_minimize_brent_quartic() {
        // Minimize f(x) = x^4 - 3x^2 + 2
        // Has local minima, find one in [0, 2]
        let result = minimize_scalar_brent(
            |x| x * x * x * x - 3.0 * x * x + 2.0,
            0.0,
            2.0,
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_brent failed");
        // Minimum should be near x = sqrt(3/2) ≈ 1.225
        let expected_x = (1.5_f64).sqrt();
        assert!((result.x - expected_x).abs() < 1e-5);
    }

    #[test]
    fn test_minimize_brent_exponential() {
        // Minimize f(x) = e^x * (x-1)^2, minimum near x=1
        let result = minimize_scalar_brent(
            |x: f64| x.exp() * (x - 1.0) * (x - 1.0),
            -1.0,
            3.0,
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_brent failed");
        assert!((result.x - 1.0).abs() < 1e-5);
        assert!(result.f_min < 1e-10);
    }

    #[test]
    fn test_minimize_bounded_simple() {
        // Minimize f(x) = (x-1)^2 with bounds (0, 3)
        let result = minimize_scalar_bounded(
            |x| (x - 1.0) * (x - 1.0),
            (0.0, 3.0),
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_bounded failed");
        assert!((result.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_minimize_bounded_boundary_minimum() {
        // Minimize f(x) = (x-0.5)^2 with bounds (0, 1), minimum at x=0.5
        let result = minimize_scalar_bounded(
            |x| (x - 0.5) * (x - 0.5),
            (0.0, 1.0),
            &ScalarOptions::default(),
        )
        .expect("minimize_scalar_bounded failed");
        assert!((result.x - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_minimize_bounded_invalid_interval() {
        let result = minimize_scalar_bounded(|x| x * x, (3.0, 1.0), &ScalarOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInterval { .. })));
    }

    #[test]
    fn test_minimize_golden_invalid_interval() {
        let result = minimize_scalar_golden(|x| x * x, 4.0, 2.0, &ScalarOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInterval { .. })));
    }

    #[test]
    fn test_minimize_brent_invalid_interval() {
        let result = minimize_scalar_brent(|x| x * x, 5.0, 1.0, &ScalarOptions::default());
        assert!(matches!(result, Err(OptimizeError::InvalidInterval { .. })));
    }

    #[test]
    fn test_minimize_convergence_comparison() {
        // Compare golden and brent on same problem
        let f = |x: f64| (x - 3.5) * (x - 3.5) + 0.1 * (x - 3.5).sin();

        let golden_result =
            minimize_scalar_golden(f, 0.0, 5.0, &ScalarOptions::default()).expect("golden failed");
        let brent_result =
            minimize_scalar_brent(f, 0.0, 5.0, &ScalarOptions::default()).expect("brent failed");

        // Both should find similar minima (allowing for different convergence)
        assert!((golden_result.x - brent_result.x).abs() < 1e-4);
        // Both should converge successfully
        assert!(brent_result.iterations > 0 && golden_result.iterations > 0);
    }
}
