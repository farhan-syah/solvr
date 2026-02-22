//! Sparse QR factorization trait.

use crate::DType;

use numr::algorithm::sparse_linalg::qr::{QrFactors, QrMetrics, QrOptions, QrSymbolic};
use numr::error::Result;
use numr::runtime::Runtime;
use numr::sparse::CscData;
use numr::tensor::Tensor;

/// High-level sparse QR factorization and solve operations.
///
/// Wraps numr's sparse QR backend functions into a unified trait-based API.
/// Supports factorization, direct solve (Ax = b), least-squares (min ||Ax - b||),
/// and symbolic/numeric split for repeated solves with the same sparsity pattern.
pub trait SparseQrAlgorithms<R: Runtime<DType = DType>> {
    /// Factorize a sparse matrix A*P = Q*R.
    ///
    /// Performs COLAMD column ordering + Householder QR.
    fn sparse_qr(&self, a: &CscData<R>, options: &QrOptions) -> Result<QrFactors<R>>;

    /// Factorize with precomputed symbolic analysis.
    ///
    /// Use when solving multiple systems with the same sparsity pattern
    /// (e.g., Newton iterations where only values change).
    fn sparse_qr_with_symbolic(
        &self,
        a: &CscData<R>,
        symbolic: &QrSymbolic,
        options: &QrOptions,
    ) -> Result<QrFactors<R>>;

    /// Factorize and return diagnostic metrics.
    fn sparse_qr_with_metrics(
        &self,
        a: &CscData<R>,
        symbolic: &QrSymbolic,
        options: &QrOptions,
    ) -> Result<(QrFactors<R>, QrMetrics)>;

    /// Compute symbolic analysis for a sparsity pattern.
    ///
    /// Reusable across numeric factorizations with the same structure.
    fn sparse_qr_symbolic(&self, a: &CscData<R>, options: &QrOptions) -> Result<QrSymbolic>;

    /// Solve Ax = b using precomputed QR factors (square full-rank systems).
    fn sparse_qr_solve(&self, factors: &QrFactors<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Solve least-squares min ||Ax - b||₂ using QR factors (overdetermined systems).
    fn sparse_least_squares(&self, factors: &QrFactors<R>, b: &Tensor<R>) -> Result<Tensor<R>>;
}
