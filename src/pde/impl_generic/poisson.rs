//! Poisson equation solver: -nabla^2 u = f on 2D grid with Dirichlet BCs.

use numr::algorithm::iterative::IterativeSolvers;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::pde::error::{PdeError, PdeResult};
use crate::pde::types::{BoundarySpec, FdmOptions, FdmResult, Grid2D};

use super::boundary::extract_boundary_values_2d;
use super::solve_sparse::solve_sparse_system;
use super::stencil::assemble_neg_laplacian_2d_dirichlet;

/// Solve the Poisson equation -nabla^2 u = f on a 2D grid.
///
/// Assembles the negative Laplacian with Dirichlet BCs as a SPD sparse matrix,
/// then solves with CG (or GMRES/BiCGSTAB based on options).
pub fn poisson_impl<R, C>(
    client: &C,
    f: &Tensor<R>,
    grid: &Grid2D,
    boundary: &[BoundarySpec<R>],
    options: &FdmOptions,
) -> PdeResult<FdmResult<R>>
where
    R: Runtime,
    C: TensorOps<R> + ScalarOps<R> + IterativeSolvers<R> + RuntimeClient<R>,
{
    let nx = grid.nx;
    let ny = grid.ny;
    let n = nx * ny;
    let device = client.device();

    if nx < 3 || ny < 3 {
        return Err(PdeError::InvalidGrid {
            context: format!("Grid must be at least 3x3, got {}x{}", nx, ny),
        });
    }

    let boundary_values = extract_boundary_values_2d(boundary, nx, ny)?;

    let f_data: Vec<f64> = f.to_vec();
    let mut rhs_data = vec![0.0; n];
    rhs_data[..n.min(f_data.len())].copy_from_slice(&f_data[..n.min(f_data.len())]);

    let a = assemble_neg_laplacian_2d_dirichlet::<R>(grid, &boundary_values, &mut rhs_data, device)
        .map_err(PdeError::from)?;

    let rhs = Tensor::<R>::from_slice(&rhs_data, &[n], device);

    let result = solve_sparse_system(client, &a, &rhs, options, "Poisson solver")?;

    Ok(FdmResult {
        solution: result.solution.reshape(&[nx, ny])?,
        iterations: result.iterations,
        residual_norm: result.residual_norm,
    })
}
