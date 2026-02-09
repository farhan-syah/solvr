//! Discrete-time LTI system traits.
//!
//! Provides algorithms for simulating discrete-time linear time-invariant systems.

use crate::signal::filter::types::{DiscreteTimeSystem, StateSpace};
use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Discrete-time LTI system algorithms.
///
/// All backends implementing DLTI simulation MUST implement this trait
/// using the EXACT SAME ALGORITHMS to ensure numerical parity.
pub trait DiscreteTimeLtiAlgorithms<R: Runtime> {
    /// Simulate output of a discrete-time linear system.
    ///
    /// # Algorithm
    ///
    /// For a state-space system:
    /// ```text
    /// x[k+1] = A·x[k] + B·u[k]
    /// y[k]   = C·x[k] + D·u[k]
    /// ```
    ///
    /// Starting from initial state x0 (or zeros), computes output y
    /// for each input sample u.
    ///
    /// # Arguments
    ///
    /// * `system` - Discrete-time LTI system (any representation)
    /// * `u` - Input signal `[n_samples]` or `[n_samples, n_inputs]`
    /// * `x0` - Initial state `[n_states]`, or None for zero state
    ///
    /// # Returns
    ///
    /// [`DlsimResult`] containing:
    /// - `t`: Time indices
    /// - `y`: Output signal
    /// - `x`: State trajectory
    fn dlsim(
        &self,
        system: &DiscreteTimeSystem<R>,
        u: &Tensor<R>,
        x0: Option<&Tensor<R>>,
        device: &R::Device,
    ) -> Result<DlsimResult<R>>;

    /// Simulate output using state-space representation directly.
    ///
    /// More efficient when system is already in state-space form.
    fn dlsim_ss(
        &self,
        ss: &StateSpace<R>,
        u: &Tensor<R>,
        x0: Option<&Tensor<R>>,
        device: &R::Device,
    ) -> Result<DlsimResult<R>>;
}

/// Result from dlsim computation.
#[derive(Debug, Clone)]
pub struct DlsimResult<R: Runtime> {
    /// Time indices (sample numbers).
    pub t: Tensor<R>,
    /// Output signal `[n_samples]` or `[n_samples, n_outputs]`.
    pub y: Tensor<R>,
    /// State trajectory `[n_samples, n_states]`.
    pub x: Tensor<R>,
}
