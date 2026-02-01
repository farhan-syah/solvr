//! Scalar optimization algorithms (1D root finding and minimization).

use super::utils::SINGULAR_THRESHOLD;
use crate::optimize::error::{OptimizeError, OptimizeResult};
use crate::optimize::scalar::{MinimizeResult, RootResult, ScalarOptions};

/// Bisection method for root finding.
pub fn bisect_impl<F>(f: F, a: f64, b: f64, options: &ScalarOptions) -> OptimizeResult<RootResult>
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

        let width = right - left;
        if width.abs() < options.tol || width.abs() / mid.abs().max(1.0) < options.rtol {
            return Ok(RootResult {
                root: mid,
                function_value: f_mid,
                iterations: iter + 1,
                bracket_width: width,
            });
        }

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

/// Brent's method for root finding.
pub fn brentq_impl<F>(f: F, a: f64, b: f64, options: &ScalarOptions) -> OptimizeResult<RootResult>
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

    let mut xa = a;
    let mut xb = b;
    let mut fa = f(xa);
    let mut fb = f(xb);

    if (fa > 0.0 && fb > 0.0) || (fa < 0.0 && fb < 0.0) {
        return Err(OptimizeError::SameSignBracket {
            fa,
            fb,
            context: "brentq".to_string(),
        });
    }

    let mut xc = xa;
    let mut fc = fa;
    let mut d = xb - xa;
    let mut e = d;

    for iter in 0..options.max_iter {
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            xc = xa;
            fc = fa;
            d = xb - xa;
            e = d;
        }

        if fc.abs() < fb.abs() {
            xa = xb;
            xb = xc;
            xc = xa;
            fa = fb;
            fb = fc;
            fc = fa;
        }

        let tol1 = 2.0 * f64::EPSILON * xb.abs() + 0.5 * options.tol;
        let xm = 0.5 * (xc - xb);

        if xm.abs() <= tol1 || fb.abs() < SINGULAR_THRESHOLD {
            return Ok(RootResult {
                root: xb,
                function_value: fb,
                iterations: iter + 1,
                bracket_width: (xc - xb).abs(),
            });
        }

        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (p, q) = if (xa - xc).abs() < SINGULAR_THRESHOLD {
                (2.0 * xm * s, 1.0 - s)
            } else {
                let q_temp = fa / fc;
                let r = fb / fc;
                (
                    s * (2.0 * xm * q_temp * (q_temp - r) - (xb - xa) * (r - 1.0)),
                    (q_temp - 1.0) * (r - 1.0) * (s - 1.0),
                )
            };

            let (p, q) = if p > 0.0 { (p, -q) } else { (-p, q) };

            if 2.0 * p < (3.0 * xm * q - (tol1 * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }

        xa = xb;
        fa = fb;

        if d.abs() > tol1 {
            xb += d;
        } else {
            xb += if xm > 0.0 { tol1 } else { -tol1 };
        }

        fb = f(xb);
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "brentq".to_string(),
    })
}

/// Newton's method for root finding.
pub fn newton_impl<F, DF>(
    f: F,
    df: DF,
    x0: f64,
    options: &ScalarOptions,
) -> OptimizeResult<RootResult>
where
    F: Fn(f64) -> f64,
    DF: Fn(f64) -> f64,
{
    let mut x = x0;
    let mut fx = f(x);

    for iter in 0..options.max_iter {
        if fx.abs() < options.tol {
            return Ok(RootResult {
                root: x,
                function_value: fx,
                iterations: iter + 1,
                bracket_width: 0.0,
            });
        }

        let dfx = df(x);
        if dfx.abs() < SINGULAR_THRESHOLD {
            return Err(OptimizeError::NumericalError {
                message: "newton: derivative too close to zero".to_string(),
            });
        }

        let x_new = x - fx / dfx;

        if (x_new - x).abs() < options.tol || (x_new - x).abs() / x.abs().max(1.0) < options.rtol {
            let fx_new = f(x_new);
            return Ok(RootResult {
                root: x_new,
                function_value: fx_new,
                iterations: iter + 1,
                bracket_width: 0.0,
            });
        }

        x = x_new;
        fx = f(x);
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "newton".to_string(),
    })
}

/// Brent's method for scalar minimization.
pub fn minimize_scalar_brent_impl<F>(
    f: F,
    bracket: Option<(f64, f64, f64)>,
    options: &ScalarOptions,
) -> OptimizeResult<MinimizeResult>
where
    F: Fn(f64) -> f64,
{
    let golden = 0.381966011250105;

    let (mut a, mut b) = if let Some((xa, _, xc)) = bracket {
        (xa.min(xc), xa.max(xc))
    } else {
        (0.0, 1.0)
    };

    let mut x = if let Some((_, xb, _)) = bracket {
        xb
    } else {
        a + golden * (b - a)
    };

    let mut w = x;
    let mut v = x;
    let mut fx = f(x);
    let mut fw = fx;
    let mut fv = fx;

    let mut d: f64 = 0.0;
    let mut e: f64 = 0.0;

    for iter in 0..options.max_iter {
        let xm = 0.5 * (a + b);
        let tol1 = options.tol * x.abs() + SINGULAR_THRESHOLD;
        let tol2 = 2.0 * tol1;

        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            return Ok(MinimizeResult {
                x,
                f_min: fx,
                iterations: iter + 1,
                bracket_width: b - a,
            });
        }

        let mut use_golden = true;

        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);

            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }

            let r_old = e;
            e = d;

            if p.abs() < (0.5 * q * r_old).abs() && p > q * (a - x) && p < q * (b - x) {
                d = p / q;
                let u = x + d;
                if (u - a) < tol2 || (b - u) < tol2 {
                    d = if x < xm { tol1 } else { -tol1 };
                }
                use_golden = false;
            }
        }

        if use_golden {
            e = if x < xm { b - x } else { a - x };
            d = golden * e;
        }

        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };

        let fu = f(u);

        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || (w - x).abs() < SINGULAR_THRESHOLD {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv
                || (v - x).abs() < SINGULAR_THRESHOLD
                || (v - w).abs() < SINGULAR_THRESHOLD
            {
                v = u;
                fv = fu;
            }
        }
    }

    Err(OptimizeError::DidNotConverge {
        iterations: options.max_iter,
        tolerance: options.tol,
        context: "minimize_scalar_brent".to_string(),
    })
}
