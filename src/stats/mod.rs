//! Statistical distributions, hypothesis tests, and descriptive statistics.
//!
//! This module provides a comprehensive statistics library with full GPU acceleration
//! via numr's multi-runtime architecture.
//!
//! # Runtime-Generic API
//!
//! Statistics algorithms are organized into focused traits:
//! - [`DescriptiveStatisticsAlgorithms`] - Computing statistics (mean, variance, skewness, etc.)
//! - [`HypothesisTestingAlgorithms`] - Statistical hypothesis tests (t-tests, ANOVA, normality)
//! - [`RegressionAlgorithms`] - Regression analysis (linear regression)
//! - [`RobustStatisticsAlgorithms`] - Robust statistics (trimmed mean, MAD, Theil-Sen)
//! - [`InformationTheoryAlgorithms`] - Information theory (entropy, KL divergence, mutual info)
//!
//! All are generic over numr's `Runtime`, so the same code works on CPU, CUDA, and WebGPU.
//!
//! # Distributions
//!
//! Distributions have both scalar and batch (tensor) methods:
//!
//! ```ignore
//! use solvr::stats::{Normal, ContinuousDistribution};
//!
//! let n = Normal::standard();
//!
//! // Scalar - for single values
//! let p = n.pdf(0.0);
//!
//! // Batch - for tensor operations (GPU-accelerated)
//! let x = client.from_slice(&[0.0, 1.0, 2.0], &device).unwrap();
//! let p_batch = n.pdf_tensor(&x, &client).unwrap();
//! ```

// Backend implementations
mod cpu;
#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "wgpu")]
mod wgpu;

// Shared generic implementations
mod helpers;
mod impl_generic;

// Traits and types
mod traits;

// Core modules
mod continuous;
mod discrete;
mod distribution;
mod error;

// Public API: Trait exports
pub use traits::{
    DescriptiveStatisticsAlgorithms, HypothesisTestingAlgorithms, InformationTheoryAlgorithms,
    LeveneCenter, LinregressResult, RegressionAlgorithms, RobustRegressionResult,
    RobustStatisticsAlgorithms, TensorDescriptiveStats, TensorTestResult, validate_stats_dtype,
};

// Public API: Distribution traits and types
pub use distribution::{ContinuousDistribution, DiscreteDistribution, Distribution};
pub use error::{StatsError, StatsResult};

// Public API: Continuous distributions
pub use continuous::{
    Beta, Cauchy, ChiSquared, Dirichlet, Exponential, FDistribution, Gamma, Gumbel, GumbelMin,
    InverseGamma, Laplace, LogNormal, Normal, Pareto, StudentT, TruncatedNormal, Uniform, Weibull,
    Wishart,
};

// Public API: Discrete distributions
pub use discrete::{
    Binomial, DiscreteUniform, Geometric, Hypergeometric, Multinomial, NegativeBinomial, Poisson,
};
