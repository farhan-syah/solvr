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
}
