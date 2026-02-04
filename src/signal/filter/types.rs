//! Filter representation types.
//!
//! This module defines the standard filter representations used in digital
//! signal processing:
//!
//! - [`TransferFunction`]: Numerator/denominator polynomial coefficients (b, a)
//! - [`ZpkFilter`]: Zeros, poles, and gain representation
//! - [`SosFilter`]: Second-order sections (cascaded biquads)
//!
//! # Coefficient Convention
//!
//! All polynomial coefficients use **descending power order** (highest power first):
//! - `H(z) = (b[0] + b[1]z^-1 + ... + b[M]z^-M) / (a[0] + a[1]z^-1 + ... + a[N]z^-N)`
//! - This matches SciPy's convention for filter functions.

use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Transfer function filter representation.
///
/// Represents a digital filter as the ratio of two polynomials:
/// ```text
/// H(z) = B(z) / A(z) = (b[0] + b[1]z^-1 + ... + b[M]z^-M)
///                    / (a[0] + a[1]z^-1 + ... + a[N]z^-N)
/// ```
///
/// # Normalization
///
/// The denominator is typically normalized so that `a[0] = 1`.
#[derive(Debug, Clone)]
pub struct TransferFunction<R: Runtime> {
    /// Numerator coefficients in descending power order.
    pub b: Tensor<R>,
    /// Denominator coefficients in descending power order.
    pub a: Tensor<R>,
}

impl<R: Runtime> TransferFunction<R> {
    /// Create a new transfer function from numerator and denominator coefficients.
    ///
    /// # Arguments
    ///
    /// * `b` - Numerator coefficients [M+1] in descending power order
    /// * `a` - Denominator coefficients [N+1] in descending power order
    pub fn new(b: Tensor<R>, a: Tensor<R>) -> Self {
        Self { b, a }
    }

    /// Get the order of the filter (max of numerator and denominator degrees).
    pub fn order(&self) -> usize {
        let num_order = self.b.shape()[0].saturating_sub(1);
        let den_order = self.a.shape()[0].saturating_sub(1);
        num_order.max(den_order)
    }

    /// Check if this is an FIR filter (denominator is just [1]).
    pub fn is_fir(&self) -> bool {
        self.a.shape()[0] == 1
    }
}

/// Zero-pole-gain filter representation.
///
/// Represents a digital filter by its zeros, poles, and gain:
/// ```text
/// H(z) = k * prod(z - zeros[i]) / prod(z - poles[i])
/// ```
///
/// # Complex Numbers
///
/// Zeros and poles are stored as separate real and imaginary tensors.
/// Complex conjugate pairs are handled automatically during conversions.
#[derive(Debug, Clone)]
pub struct ZpkFilter<R: Runtime> {
    /// Real parts of zeros [num_zeros].
    pub zeros_real: Tensor<R>,
    /// Imaginary parts of zeros [num_zeros].
    pub zeros_imag: Tensor<R>,
    /// Real parts of poles [num_poles].
    pub poles_real: Tensor<R>,
    /// Imaginary parts of poles [num_poles].
    pub poles_imag: Tensor<R>,
    /// System gain.
    pub gain: f64,
}

impl<R: Runtime> ZpkFilter<R> {
    /// Create a new ZPK filter.
    pub fn new(
        zeros_real: Tensor<R>,
        zeros_imag: Tensor<R>,
        poles_real: Tensor<R>,
        poles_imag: Tensor<R>,
        gain: f64,
    ) -> Self {
        Self {
            zeros_real,
            zeros_imag,
            poles_real,
            poles_imag,
            gain,
        }
    }

    /// Get the number of zeros.
    pub fn num_zeros(&self) -> usize {
        self.zeros_real.shape()[0]
    }

    /// Get the number of poles.
    pub fn num_poles(&self) -> usize {
        self.poles_real.shape()[0]
    }
}

/// Second-order sections filter representation.
///
/// Represents a digital filter as a cascade of second-order (biquad) sections:
/// ```text
/// H(z) = prod_i H_i(z)
///
/// H_i(z) = (b0_i + b1_i*z^-1 + b2_i*z^-2) / (1 + a1_i*z^-1 + a2_i*z^-2)
/// ```
///
/// # Section Format
///
/// Each section is stored as 6 coefficients: `[b0, b1, b2, a0, a1, a2]`
/// where `a0` is typically 1.0 (normalized).
///
/// # Advantages
///
/// SOS representation is more numerically stable than transfer function
/// for high-order filters, as it avoids computing high-degree polynomials.
#[derive(Debug, Clone)]
pub struct SosFilter<R: Runtime> {
    /// Second-order sections [num_sections, 6].
    /// Each row: [b0, b1, b2, a0, a1, a2]
    pub sections: Tensor<R>,
}

impl<R: Runtime> SosFilter<R> {
    /// Create a new SOS filter from sections tensor.
    ///
    /// # Arguments
    ///
    /// * `sections` - Tensor of shape [num_sections, 6]
    pub fn new(sections: Tensor<R>) -> Self {
        Self { sections }
    }

    /// Get the number of second-order sections.
    pub fn num_sections(&self) -> usize {
        self.sections.shape()[0]
    }

    /// Get the overall filter order (2 * num_sections).
    pub fn order(&self) -> usize {
        2 * self.num_sections()
    }
}

/// Filter type for design functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterType {
    /// Lowpass filter - passes frequencies below cutoff.
    Lowpass,
    /// Highpass filter - passes frequencies above cutoff.
    Highpass,
    /// Bandpass filter - passes frequencies between two cutoffs.
    Bandpass,
    /// Bandstop (notch) filter - rejects frequencies between two cutoffs.
    Bandstop,
}

/// Filter output format for design functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FilterOutput {
    /// Return as transfer function coefficients (b, a).
    #[default]
    Ba,
    /// Return as zeros, poles, gain.
    Zpk,
    /// Return as second-order sections.
    Sos,
}

/// Analog filter prototype specification.
///
/// Contains the poles and zeros of an analog prototype filter
/// (lowpass with cutoff at 1 rad/s).
#[derive(Debug, Clone)]
pub struct AnalogPrototype<R: Runtime> {
    /// Real parts of analog zeros.
    pub zeros_real: Tensor<R>,
    /// Imaginary parts of analog zeros.
    pub zeros_imag: Tensor<R>,
    /// Real parts of analog poles.
    pub poles_real: Tensor<R>,
    /// Imaginary parts of analog poles.
    pub poles_imag: Tensor<R>,
    /// System gain.
    pub gain: f64,
}

impl<R: Runtime> AnalogPrototype<R> {
    /// Create a new analog prototype.
    pub fn new(
        zeros_real: Tensor<R>,
        zeros_imag: Tensor<R>,
        poles_real: Tensor<R>,
        poles_imag: Tensor<R>,
        gain: f64,
    ) -> Self {
        Self {
            zeros_real,
            zeros_imag,
            poles_real,
            poles_imag,
            gain,
        }
    }
}

/// State-space representation of a linear time-invariant system.
///
/// Represents a system in state-space form:
/// ```text
/// ẋ(t) = A·x(t) + B·u(t)
/// y(t) = C·x(t) + D·u(t)
/// ```
///
/// For discrete-time systems:
/// ```text
/// x[k+1] = A·x[k] + B·u[k]
/// y[k]   = C·x[k] + D·u[k]
/// ```
///
/// # Matrix Dimensions
///
/// For a system with n states, m inputs, and p outputs:
/// - A: n × n (state matrix)
/// - B: n × m (input matrix)
/// - C: p × n (output matrix)
/// - D: p × m (feedthrough matrix)
#[derive(Debug, Clone)]
pub struct StateSpace<R: Runtime> {
    /// State matrix (n × n).
    pub a: Tensor<R>,
    /// Input matrix (n × m).
    pub b: Tensor<R>,
    /// Output matrix (p × n).
    pub c: Tensor<R>,
    /// Feedthrough matrix (p × m).
    pub d: Tensor<R>,
}

impl<R: Runtime> StateSpace<R> {
    /// Create a new state-space system.
    ///
    /// # Arguments
    ///
    /// * `a` - State matrix (n × n)
    /// * `b` - Input matrix (n × m)
    /// * `c` - Output matrix (p × n)
    /// * `d` - Feedthrough matrix (p × m)
    pub fn new(a: Tensor<R>, b: Tensor<R>, c: Tensor<R>, d: Tensor<R>) -> Self {
        Self { a, b, c, d }
    }

    /// Get the number of states.
    pub fn num_states(&self) -> usize {
        self.a.shape()[0]
    }

    /// Get the number of inputs.
    pub fn num_inputs(&self) -> usize {
        if self.b.ndim() == 1 {
            1
        } else {
            self.b.shape()[1]
        }
    }

    /// Get the number of outputs.
    pub fn num_outputs(&self) -> usize {
        if self.c.ndim() == 1 {
            1
        } else {
            self.c.shape()[0]
        }
    }
}

/// Discrete-time linear time-invariant system.
///
/// Wraps a system representation with a sampling time for discrete-time analysis.
#[derive(Debug, Clone)]
pub struct DiscreteTimeSystem<R: Runtime> {
    /// System representation.
    pub system: SystemRepresentation<R>,
    /// Sampling time (seconds). None for continuous-time.
    pub dt: Option<f64>,
}

impl<R: Runtime> DiscreteTimeSystem<R> {
    /// Create a new discrete-time system.
    pub fn new(system: SystemRepresentation<R>, dt: Option<f64>) -> Self {
        Self { system, dt }
    }

    /// Check if this is a discrete-time system.
    pub fn is_discrete(&self) -> bool {
        self.dt.is_some()
    }
}

/// System representation enum.
///
/// Allows a system to be represented in any of the standard forms.
#[derive(Debug, Clone)]
pub enum SystemRepresentation<R: Runtime> {
    /// Transfer function (numerator, denominator polynomials).
    TransferFunction(TransferFunction<R>),
    /// Zero-pole-gain representation.
    ZeroPoleGain(ZpkFilter<R>),
    /// State-space representation.
    StateSpace(StateSpace<R>),
}
