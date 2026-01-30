//! Types for ODE solvers.

/// ODE solver method.
///
/// # Available Methods
///
/// | Method | Order | Stages | Use Case |
/// |--------|-------|--------|----------|
/// | RK23   | 2(3)  | 4      | Fast, lower accuracy |
/// | RK45   | 4(5)  | 6      | General purpose (recommended) |
/// | DOP853 | 8(5,3)| 12     | High accuracy requirements |
///
/// # Choosing a Method
///
/// - **RK23**: Use when speed is more important than accuracy, or for getting
///   a rough initial estimate.
/// - **RK45**: The default choice. Works well for most non-stiff problems.
/// - **DOP853**: Use for high-accuracy requirements on smooth problems.
///   Fewer steps than RK45 for tight tolerances, but more work per step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ODEMethod {
    /// Bogacki-Shampine 2(3) - low accuracy, fast.
    ///
    /// 4 stages per step. Good for problems where speed matters more than
    /// precision, or for getting a rough initial solution.
    RK23,

    /// Dormand-Prince 4(5) - general purpose (default).
    ///
    /// 6 stages per step. Recommended for most problems. Good balance of
    /// accuracy and speed. Uses FSAL (First Same As Last) property for
    /// efficiency.
    #[default]
    RK45,

    /// Dormand-Prince 8(5,3) - high accuracy.
    ///
    /// 12 stages per step. An 8th order method with embedded 5th order error
    /// estimator. Best for high-accuracy requirements on smooth problems.
    /// Takes larger steps than RK45 for tight tolerances, compensating for
    /// the additional work per step.
    DOP853,
}

impl ODEMethod {
    /// Get the order of the method.
    pub fn order(&self) -> usize {
        match self {
            Self::RK23 => 3,
            Self::RK45 => 5,
            Self::DOP853 => 8,
        }
    }

    /// Get the error estimator order.
    pub fn error_order(&self) -> usize {
        match self {
            Self::RK23 => 2,
            Self::RK45 => 4,
            Self::DOP853 => 5,
        }
    }
}

/// Options for ODE solvers.
#[derive(Debug, Clone)]
pub struct ODEOptions {
    /// Solver method (default: RK45)
    pub method: ODEMethod,

    /// Relative tolerance (default: 1e-3)
    pub rtol: f64,

    /// Absolute tolerance (default: 1e-6)
    pub atol: f64,

    /// Initial step size (default: auto-computed)
    pub h0: Option<f64>,

    /// Maximum step size (default: unbounded)
    pub max_step: Option<f64>,

    /// Minimum step size (default: machine epsilon)
    pub min_step: Option<f64>,

    /// Maximum number of steps (default: 10000)
    pub max_steps: usize,

    /// Dense output - evaluate solution at any point (default: false)
    pub dense_output: bool,
}

impl Default for ODEOptions {
    fn default() -> Self {
        Self {
            method: ODEMethod::default(),
            rtol: 1e-3,
            atol: 1e-6,
            h0: None,
            max_step: None,
            min_step: None,
            max_steps: 10000,
            dense_output: false,
        }
    }
}

impl ODEOptions {
    /// Create options with specified tolerances.
    pub fn with_tolerances(rtol: f64, atol: f64) -> Self {
        Self {
            rtol,
            atol,
            ..Default::default()
        }
    }

    /// Create options with specified method.
    pub fn with_method(method: ODEMethod) -> Self {
        Self {
            method,
            ..Default::default()
        }
    }

    /// Set the method.
    pub fn method(mut self, method: ODEMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the tolerances.
    pub fn tolerances(mut self, rtol: f64, atol: f64) -> Self {
        self.rtol = rtol;
        self.atol = atol;
        self
    }

    /// Set the initial step size.
    pub fn initial_step(mut self, h0: f64) -> Self {
        self.h0 = Some(h0);
        self
    }

    /// Set step size bounds.
    pub fn step_bounds(mut self, min: f64, max: f64) -> Self {
        self.min_step = Some(min);
        self.max_step = Some(max);
        self
    }

    /// Set maximum number of steps.
    pub fn max_steps(mut self, n: usize) -> Self {
        self.max_steps = n;
        self
    }
}

/// Result of ODE integration.
#[derive(Debug, Clone)]
pub struct ODEResult {
    /// Time points where solution was computed
    pub t: Vec<f64>,

    /// Solution values at each time point (y[i] is solution at t[i])
    pub y: Vec<Vec<f64>>,

    /// Whether integration was successful
    pub success: bool,

    /// Status message (e.g., why integration failed)
    pub message: Option<String>,

    /// Number of function evaluations
    pub nfev: usize,

    /// Number of accepted steps
    pub naccept: usize,

    /// Number of rejected steps
    pub nreject: usize,

    /// Method used for integration
    pub method: ODEMethod,
}

impl ODEResult {
    /// Get the final time.
    pub fn t_final(&self) -> f64 {
        *self.t.last().unwrap_or(&0.0)
    }

    /// Get the final state.
    pub fn y_final(&self) -> &[f64] {
        self.y.last().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Total number of steps (accepted + rejected).
    pub fn nsteps(&self) -> usize {
        self.naccept + self.nreject
    }
}

/// Dense solution for interpolating between computed points.
///
/// This allows evaluating the solution at arbitrary times within
/// the integration span.
#[derive(Debug, Clone)]
pub struct ODESolution {
    /// Computed time points
    t: Vec<f64>,
    /// Computed solution values
    y: Vec<Vec<f64>>,
    /// Hermite interpolation coefficients (for dense output)
    hermite: Option<Vec<HermiteCoeffs>>,
}

/// Hermite interpolation coefficients for one step.
#[derive(Debug, Clone)]
struct HermiteCoeffs {
    /// Start time
    t0: f64,
    /// End time
    t1: f64,
    /// State at t0
    y0: Vec<f64>,
    /// State at t1
    y1: Vec<f64>,
    /// Derivative at t0
    f0: Vec<f64>,
    /// Derivative at t1
    f1: Vec<f64>,
}

impl ODESolution {
    /// Create a new dense solution from integration results.
    pub fn new(t: Vec<f64>, y: Vec<Vec<f64>>) -> Self {
        Self {
            t,
            y,
            hermite: None,
        }
    }

    /// Create a dense solution with Hermite interpolation data.
    pub fn with_hermite(t: Vec<f64>, y: Vec<Vec<f64>>, derivatives: Vec<Vec<f64>>) -> Self {
        let n = t.len();
        if n < 2 {
            return Self::new(t, y);
        }

        let mut hermite = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            hermite.push(HermiteCoeffs {
                t0: t[i],
                t1: t[i + 1],
                y0: y[i].clone(),
                y1: y[i + 1].clone(),
                f0: derivatives[i].clone(),
                f1: derivatives[i + 1].clone(),
            });
        }

        Self {
            t,
            y,
            hermite: Some(hermite),
        }
    }

    /// Evaluate the solution at time `t`.
    ///
    /// Returns `None` if `t` is outside the integration span or if
    /// the solution data is invalid.
    pub fn eval(&self, t: f64) -> Option<Vec<f64>> {
        // Safe access to bounds - returns None if empty
        let (&t_min, &t_max) = self.t.first().zip(self.t.last())?;

        if t < t_min || t > t_max {
            return None;
        }

        // Find the interval containing t using binary search for efficiency
        let idx = match self
            .t
            .binary_search_by(|probe| probe.partial_cmp(&t).unwrap_or(std::cmp::Ordering::Equal))
        {
            Ok(i) => i.min(self.t.len().saturating_sub(2)),
            Err(i) => i.saturating_sub(1),
        };

        // Use Hermite interpolation if available
        if let Some(coeffs) = self.hermite.as_ref().and_then(|h| h.get(idx)) {
            return Some(self.hermite_interp(coeffs, t));
        }

        // Fall back to linear interpolation
        // Safe access: get both points, return None if indices invalid
        let y0 = self.y.get(idx)?;
        let y1 = self.y.get(idx + 1).or_else(|| self.y.last())?;
        let t0 = self.t.get(idx)?;
        let t1 = self.t.get(idx + 1).unwrap_or(&t_max);

        let tau = if (t1 - t0).abs() < f64::EPSILON {
            0.0
        } else {
            (t - t0) / (t1 - t0)
        };

        let result: Vec<f64> = y0
            .iter()
            .zip(y1.iter())
            .map(|(&a, &b)| a + tau * (b - a))
            .collect();

        Some(result)
    }

    /// Hermite cubic interpolation.
    fn hermite_interp(&self, coeffs: &HermiteCoeffs, t: f64) -> Vec<f64> {
        let h = coeffs.t1 - coeffs.t0;
        let tau = (t - coeffs.t0) / h;
        let tau2 = tau * tau;
        let tau3 = tau2 * tau;

        // Hermite basis functions
        let h00 = 2.0 * tau3 - 3.0 * tau2 + 1.0;
        let h10 = tau3 - 2.0 * tau2 + tau;
        let h01 = -2.0 * tau3 + 3.0 * tau2;
        let h11 = tau3 - tau2;

        coeffs
            .y0
            .iter()
            .zip(&coeffs.y1)
            .zip(&coeffs.f0)
            .zip(&coeffs.f1)
            .map(|(((&y0, &y1), &f0), &f1)| h00 * y0 + h10 * h * f0 + h01 * y1 + h11 * h * f1)
            .collect()
    }

    /// Get the time span.
    pub fn t_span(&self) -> Option<(f64, f64)> {
        // Safe access using Option combinators - no unwrap needed
        self.t.first().zip(self.t.last()).map(|(&a, &b)| (a, b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ode_method() {
        assert_eq!(ODEMethod::RK23.order(), 3);
        assert_eq!(ODEMethod::RK23.error_order(), 2);
        assert_eq!(ODEMethod::RK45.order(), 5);
        assert_eq!(ODEMethod::RK45.error_order(), 4);
        assert_eq!(ODEMethod::DOP853.order(), 8);
        assert_eq!(ODEMethod::DOP853.error_order(), 5);
    }

    #[test]
    fn test_ode_options() {
        let opts = ODEOptions::default();
        assert_eq!(opts.method, ODEMethod::RK45);
        assert_eq!(opts.rtol, 1e-3);
        assert_eq!(opts.atol, 1e-6);

        let opts = ODEOptions::with_tolerances(1e-6, 1e-9);
        assert_eq!(opts.rtol, 1e-6);
        assert_eq!(opts.atol, 1e-9);
    }

    #[test]
    fn test_ode_solution_linear_interp() {
        let t = vec![0.0, 1.0, 2.0];
        let y = vec![vec![0.0], vec![1.0], vec![2.0]];

        let sol = ODESolution::new(t, y);

        // Exact points
        let y0 = sol.eval(0.0).unwrap();
        assert!((y0[0] - 0.0).abs() < 1e-10);

        let y1 = sol.eval(1.0).unwrap();
        assert!((y1[0] - 1.0).abs() < 1e-10);

        // Interpolated points
        let y_mid = sol.eval(0.5).unwrap();
        assert!((y_mid[0] - 0.5).abs() < 1e-10);

        // Outside range
        assert!(sol.eval(-0.1).is_none());
        assert!(sol.eval(2.1).is_none());
    }

    #[test]
    fn test_ode_solution_hermite_interp() {
        // y = x^2, y' = 2x
        let t = vec![0.0, 1.0];
        let y = vec![vec![0.0], vec![1.0]];
        let f = vec![vec![0.0], vec![2.0]]; // derivatives

        let sol = ODESolution::with_hermite(t, y, f);

        // At x = 0.5, y should be close to 0.25
        let y_mid = sol.eval(0.5).unwrap();
        assert!((y_mid[0] - 0.25).abs() < 0.1);
    }
}
