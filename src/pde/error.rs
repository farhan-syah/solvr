//! Error types for PDE solver operations.

use std::fmt;

/// Result type for PDE operations.
pub type PdeResult<T> = Result<T, PdeError>;

/// Errors that can occur during PDE solving.
#[derive(Debug, Clone)]
pub enum PdeError {
    /// Invalid grid specification.
    InvalidGrid { context: String },

    /// Invalid boundary condition specification.
    InvalidBoundary { context: String },

    /// Iterative solver did not converge.
    DidNotConverge {
        iterations: usize,
        tolerance: f64,
        context: String,
    },

    /// Invalid parameter value.
    InvalidParameter { parameter: String, message: String },

    /// Singular or ill-conditioned system.
    SingularSystem { context: String },

    /// Error from underlying numr operation.
    NumrError(String),

    /// Error from solvr integrate module.
    IntegrateError(String),
}

impl fmt::Display for PdeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGrid { context } => write!(f, "Invalid grid: {}", context),
            Self::InvalidBoundary { context } => write!(f, "Invalid boundary: {}", context),
            Self::DidNotConverge {
                iterations,
                tolerance,
                context,
            } => write!(
                f,
                "{}: did not converge after {} iterations (tolerance: {:.2e})",
                context, iterations, tolerance
            ),
            Self::InvalidParameter { parameter, message } => {
                write!(f, "Invalid parameter '{}': {}", parameter, message)
            }
            Self::SingularSystem { context } => write!(f, "Singular system: {}", context),
            Self::NumrError(msg) => write!(f, "numr error: {}", msg),
            Self::IntegrateError(msg) => write!(f, "integrate error: {}", msg),
        }
    }
}

impl std::error::Error for PdeError {}

impl From<numr::error::Error> for PdeError {
    fn from(err: numr::error::Error) -> Self {
        Self::NumrError(err.to_string())
    }
}

impl From<crate::integrate::IntegrateError> for PdeError {
    fn from(err: crate::integrate::IntegrateError) -> Self {
        Self::IntegrateError(err.to_string())
    }
}
