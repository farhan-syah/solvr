//! Shared sparse iterative solver dispatch.
use crate::DType;

use numr::algorithm::iterative::{
    BiCgStabOptions, CgOptions, GmresOptions, IterativeSolvers, PreconditionerType,
};
use numr::runtime::Runtime;
use numr::sparse::CsrData;
use numr::tensor::Tensor;

use crate::pde::error::{PdeError, PdeResult};
use crate::pde::types::{FdmOptions, Preconditioner, SparseSolver};

/// Result from sparse solve dispatch.
pub struct SparsesolveResult<R: Runtime<DType = DType>> {
    pub solution: Tensor<R>,
    pub iterations: usize,
    pub residual_norm: f64,
}

/// Dispatch to CG/GMRES/BiCGSTAB based on `FdmOptions`, check convergence.
pub fn solve_sparse_system<R, C>(
    client: &C,
    a: &CsrData<R>,
    rhs: &Tensor<R>,
    options: &FdmOptions,
    context: &str,
) -> PdeResult<SparsesolveResult<R>>
where
    R: Runtime<DType = DType>,
    C: IterativeSolvers<R>,
{
    let preconditioner = match options.preconditioner {
        Preconditioner::None => PreconditionerType::None,
        Preconditioner::Ilu0 => PreconditionerType::Ilu0,
        Preconditioner::Ic0 => PreconditionerType::Ic0,
    };

    let (solution, iterations, residual_norm, converged) = match options.solver {
        SparseSolver::Cg => {
            let opts = CgOptions {
                max_iter: options.max_iter,
                rtol: options.tolerance,
                atol: options.tolerance * 0.01,
                preconditioner,
            };
            let r = client.cg(a, rhs, None, opts).map_err(PdeError::from)?;
            (r.solution, r.iterations, r.residual_norm, r.converged)
        }
        SparseSolver::Gmres => {
            let opts = GmresOptions {
                max_iter: options.max_iter,
                rtol: options.tolerance,
                atol: options.tolerance * 0.01,
                restart: 30,
                preconditioner,
                ..Default::default()
            };
            let r = client.gmres(a, rhs, None, opts).map_err(PdeError::from)?;
            (r.solution, r.iterations, r.residual_norm, r.converged)
        }
        SparseSolver::BiCgStab => {
            let opts = BiCgStabOptions {
                max_iter: options.max_iter,
                rtol: options.tolerance,
                atol: options.tolerance * 0.01,
                preconditioner,
            };
            let r = client
                .bicgstab(a, rhs, None, opts)
                .map_err(PdeError::from)?;
            (r.solution, r.iterations, r.residual_norm, r.converged)
        }
    };

    if !converged {
        return Err(PdeError::DidNotConverge {
            iterations,
            tolerance: options.tolerance,
            context: context.to_string(),
        });
    }

    Ok(SparsesolveResult {
        solution,
        iterations,
        residual_norm,
    })
}
