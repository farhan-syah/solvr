//! Sparse stencil assembly for finite difference methods.
//!
//! Builds sparse Laplacian matrices from grid specifications.
//! Stencil assembly is a one-time setup cost at API boundary.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::sparse::CsrData;
use numr::tensor::Tensor;

use crate::pde::types::{Grid2D, Grid3D};

/// Assemble a 2D negative Laplacian (-nabla^2) with Dirichlet BCs as CSR.
///
/// Boundary rows become identity rows. Interior neighbors that are boundary
/// nodes have their contribution moved to the RHS vector.
///
/// This solves: -nabla^2 u = f, so the matrix is SPD for interior nodes.
pub fn assemble_neg_laplacian_2d_dirichlet<R>(
    grid: &Grid2D,
    boundary_values: &[f64],
    rhs_data: &mut [f64],
    device: &R::Device,
) -> Result<CsrData<R>>
where
    R: Runtime,
{
    let nx = grid.nx;
    let ny = grid.ny;
    let n = nx * ny;
    let dx2 = grid.dx * grid.dx;
    let dy2 = grid.dy * grid.dy;

    let is_boundary =
        |i: usize, j: usize| -> bool { i == 0 || i == nx - 1 || j == 0 || j == ny - 1 };

    let bval = |idx: usize| -> f64 {
        if idx < boundary_values.len() {
            boundary_values[idx]
        } else {
            0.0
        }
    };

    let mut rows = Vec::with_capacity(5 * n);
    let mut cols = Vec::with_capacity(5 * n);
    let mut vals: Vec<f64> = Vec::with_capacity(5 * n);

    for i in 0..nx {
        for j in 0..ny {
            let idx = i * ny + j;

            if is_boundary(i, j) {
                rows.push(idx as i64);
                cols.push(idx as i64);
                vals.push(1.0);
                rhs_data[idx] = bval(idx);
            } else {
                let mut center = 0.0;

                // Left (i-1, j)
                let left_idx = (i - 1) * ny + j;
                if is_boundary(i - 1, j) {
                    rhs_data[idx] += bval(left_idx) / dx2;
                } else {
                    rows.push(idx as i64);
                    cols.push(left_idx as i64);
                    vals.push(-1.0 / dx2);
                }
                center += 1.0 / dx2;

                // Right (i+1, j)
                let right_idx = (i + 1) * ny + j;
                if is_boundary(i + 1, j) {
                    rhs_data[idx] += bval(right_idx) / dx2;
                } else {
                    rows.push(idx as i64);
                    cols.push(right_idx as i64);
                    vals.push(-1.0 / dx2);
                }
                center += 1.0 / dx2;

                // Bottom (i, j-1)
                let bottom_idx = i * ny + (j - 1);
                if is_boundary(i, j - 1) {
                    rhs_data[idx] += bval(bottom_idx) / dy2;
                } else {
                    rows.push(idx as i64);
                    cols.push(bottom_idx as i64);
                    vals.push(-1.0 / dy2);
                }
                center += 1.0 / dy2;

                // Top (i, j+1)
                let top_idx = i * ny + (j + 1);
                if is_boundary(i, j + 1) {
                    rhs_data[idx] += bval(top_idx) / dy2;
                } else {
                    rows.push(idx as i64);
                    cols.push(top_idx as i64);
                    vals.push(-1.0 / dy2);
                }
                center += 1.0 / dy2;

                rows.push(idx as i64);
                cols.push(idx as i64);
                vals.push(center);
            }
        }
    }

    let nnz = rows.len();
    let row_t = Tensor::<R>::from_slice(&rows, &[nnz], device);
    let col_t = Tensor::<R>::from_slice(&cols, &[nnz], device);
    let val_t = Tensor::<R>::from_slice(&vals, &[nnz], device);

    // COO to CSR: sort by row, build row_ptrs
    coo_to_csr_sorted::<R>(&row_t, &col_t, &val_t, [n, n], device)
}

/// Assemble 2D Laplacian (not negated) as CSR for time-dependent solvers.
///
/// This returns L such that L*u approximates nabla^2 u.
/// Used by heat/wave equation solvers where the sign is handled externally.
pub fn assemble_laplacian_2d<R>(grid: &Grid2D, device: &R::Device) -> Result<CsrData<R>>
where
    R: Runtime,
{
    let nx = grid.nx;
    let ny = grid.ny;
    let n = nx * ny;
    let dx2 = grid.dx * grid.dx;
    let dy2 = grid.dy * grid.dy;

    let mut rows = Vec::with_capacity(5 * n);
    let mut cols = Vec::with_capacity(5 * n);
    let mut vals: Vec<f64> = Vec::with_capacity(5 * n);

    for i in 0..nx {
        for j in 0..ny {
            let idx = i * ny + j;
            let mut center = 0.0;

            if i > 0 {
                rows.push(idx as i64);
                cols.push((idx - ny) as i64);
                vals.push(1.0 / dx2);
                center -= 1.0 / dx2;
            }
            if i < nx - 1 {
                rows.push(idx as i64);
                cols.push((idx + ny) as i64);
                vals.push(1.0 / dx2);
                center -= 1.0 / dx2;
            }
            if j > 0 {
                rows.push(idx as i64);
                cols.push((idx - 1) as i64);
                vals.push(1.0 / dy2);
                center -= 1.0 / dy2;
            }
            if j < ny - 1 {
                rows.push(idx as i64);
                cols.push((idx + 1) as i64);
                vals.push(1.0 / dy2);
                center -= 1.0 / dy2;
            }

            rows.push(idx as i64);
            cols.push(idx as i64);
            vals.push(center);
        }
    }

    let nnz = rows.len();
    let row_t = Tensor::<R>::from_slice(&rows, &[nnz], device);
    let col_t = Tensor::<R>::from_slice(&cols, &[nnz], device);
    let val_t = Tensor::<R>::from_slice(&vals, &[nnz], device);

    coo_to_csr_sorted::<R>(&row_t, &col_t, &val_t, [n, n], device)
}

/// Assemble 3D Laplacian as CSR (7-point stencil).
pub fn assemble_laplacian_3d<R>(grid: &Grid3D, device: &R::Device) -> Result<CsrData<R>>
where
    R: Runtime,
{
    let nx = grid.nx;
    let ny = grid.ny;
    let nz = grid.nz;
    let n = nx * ny * nz;
    let nyz = ny * nz;
    let dx2 = grid.dx * grid.dx;
    let dy2 = grid.dy * grid.dy;
    let dz2 = grid.dz * grid.dz;

    let mut rows = Vec::with_capacity(7 * n);
    let mut cols = Vec::with_capacity(7 * n);
    let mut vals: Vec<f64> = Vec::with_capacity(7 * n);

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let idx = i * nyz + j * nz + k;
                let mut center = 0.0;

                if i > 0 {
                    rows.push(idx as i64);
                    cols.push((idx - nyz) as i64);
                    vals.push(1.0 / dx2);
                    center -= 1.0 / dx2;
                }
                if i < nx - 1 {
                    rows.push(idx as i64);
                    cols.push((idx + nyz) as i64);
                    vals.push(1.0 / dx2);
                    center -= 1.0 / dx2;
                }
                if j > 0 {
                    rows.push(idx as i64);
                    cols.push((idx - nz) as i64);
                    vals.push(1.0 / dy2);
                    center -= 1.0 / dy2;
                }
                if j < ny - 1 {
                    rows.push(idx as i64);
                    cols.push((idx + nz) as i64);
                    vals.push(1.0 / dy2);
                    center -= 1.0 / dy2;
                }
                if k > 0 {
                    rows.push(idx as i64);
                    cols.push((idx - 1) as i64);
                    vals.push(1.0 / dz2);
                    center -= 1.0 / dz2;
                }
                if k < nz - 1 {
                    rows.push(idx as i64);
                    cols.push((idx + 1) as i64);
                    vals.push(1.0 / dz2);
                    center -= 1.0 / dz2;
                }

                rows.push(idx as i64);
                cols.push(idx as i64);
                vals.push(center);
            }
        }
    }

    let nnz = rows.len();
    let row_t = Tensor::<R>::from_slice(&rows, &[nnz], device);
    let col_t = Tensor::<R>::from_slice(&cols, &[nnz], device);
    let val_t = Tensor::<R>::from_slice(&vals, &[nnz], device);

    coo_to_csr_sorted::<R>(&row_t, &col_t, &val_t, [n, n], device)
}

/// Convert already-sorted COO data to CSR format.
///
/// The stencil assembly produces entries sorted by row (and within each row by column),
/// so we can build row_ptrs directly without a general sort.
fn coo_to_csr_sorted<R: Runtime>(
    row_indices: &Tensor<R>,
    col_indices: &Tensor<R>,
    values: &Tensor<R>,
    shape: [usize; 2],
    device: &R::Device,
) -> Result<CsrData<R>> {
    let [nrows, _ncols] = shape;
    let rows_vec: Vec<i64> = row_indices.to_vec();
    let nnz = rows_vec.len();

    // Build row_ptrs
    let mut row_ptrs = vec![0i64; nrows + 1];
    for &r in &rows_vec {
        row_ptrs[r as usize + 1] += 1;
    }
    // Cumulative sum
    for i in 1..=nrows {
        row_ptrs[i] += row_ptrs[i - 1];
    }
    debug_assert_eq!(row_ptrs[nrows] as usize, nnz);

    let rp_tensor = Tensor::<R>::from_slice(&row_ptrs, &[nrows + 1], device);
    CsrData::new(rp_tensor, col_indices.clone(), values.clone(), shape)
}
