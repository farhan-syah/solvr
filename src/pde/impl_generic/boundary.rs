//! Shared boundary condition extraction helpers.
use crate::DType;

use numr::runtime::Runtime;

use crate::pde::error::{PdeError, PdeResult};
use crate::pde::types::{BoundaryCondition, BoundarySide, BoundarySpec};

/// Extract left/right Dirichlet scalar values from 1D boundary specs.
///
/// Returns `(left, right)` values. Only Dirichlet BCs are supported;
/// Neumann and Periodic produce an error with the given `solver_name` context.
pub fn extract_dirichlet_1d_bcs<R: Runtime<DType = DType>>(
    boundary: &[BoundarySpec<R>],
    solver_name: &str,
) -> PdeResult<(f64, f64)> {
    let mut left = 0.0;
    let mut right = 0.0;

    for spec in boundary {
        match &spec.condition {
            BoundaryCondition::Dirichlet(vals) => {
                let v: Vec<f64> = vals.to_vec();
                let val = if v.is_empty() { 0.0 } else { v[0] };
                match spec.side {
                    BoundarySide::Left => left = val,
                    BoundarySide::Right => right = val,
                    BoundarySide::All => {
                        left = val;
                        right = val;
                    }
                    _ => {}
                }
            }
            _ => {
                return Err(PdeError::InvalidBoundary {
                    context: format!("Only Dirichlet BCs supported for {}", solver_name),
                });
            }
        }
    }

    Ok((left, right))
}

/// Extract a single Dirichlet scalar from the first boundary spec.
///
/// Used by solvers that apply a uniform value to all boundary nodes.
pub fn extract_dirichlet_scalar<R: Runtime<DType = DType>>(
    boundary: &[BoundarySpec<R>],
    solver_name: &str,
) -> PdeResult<f64> {
    if let Some(spec) = boundary.first() {
        match &spec.condition {
            BoundaryCondition::Dirichlet(vals) => {
                let v: Vec<f64> = vals.to_vec();
                Ok(if v.is_empty() { 0.0 } else { v[0] })
            }
            _ => Err(PdeError::InvalidBoundary {
                context: format!("Only Dirichlet BCs supported for {}", solver_name),
            }),
        }
    } else {
        Ok(0.0)
    }
}

/// Extract 2D boundary values as a flat array `[nx*ny]` from BoundarySpec.
///
/// Handles per-side Dirichlet values for 2D grids with row-major layout.
pub fn extract_boundary_values_2d<R: Runtime<DType = DType>>(
    boundary: &[BoundarySpec<R>],
    nx: usize,
    ny: usize,
) -> PdeResult<Vec<f64>> {
    let n = nx * ny;
    let mut values = vec![0.0; n];

    for spec in boundary {
        match &spec.condition {
            BoundaryCondition::Dirichlet(vals) => {
                let v: Vec<f64> = vals.to_vec();
                match spec.side {
                    BoundarySide::Left => {
                        let copy_len = ny.min(v.len());
                        values[..copy_len].copy_from_slice(&v[..copy_len]);
                    }
                    BoundarySide::Right => {
                        let copy_len = ny.min(v.len());
                        for (idx, &val) in v[..copy_len].iter().enumerate() {
                            values[(nx - 1) * ny + idx] = val;
                        }
                    }
                    BoundarySide::Bottom => {
                        let copy_len = nx.min(v.len());
                        for (i, &val) in v[..copy_len].iter().enumerate() {
                            values[i * ny] = val;
                        }
                    }
                    BoundarySide::Top => {
                        let copy_len = nx.min(v.len());
                        for (i, &val) in v[..copy_len].iter().enumerate() {
                            values[i * ny + (ny - 1)] = val;
                        }
                    }
                    BoundarySide::All => {
                        let val = if v.is_empty() { 0.0 } else { v[0] };
                        for i in 0..nx {
                            for j in 0..ny {
                                if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 {
                                    values[i * ny + j] = val;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            BoundaryCondition::Neumann(_) => {
                return Err(PdeError::InvalidBoundary {
                    context: "Neumann BCs not yet supported for Poisson solver".to_string(),
                });
            }
            BoundaryCondition::Periodic => {
                return Err(PdeError::InvalidBoundary {
                    context: "Periodic BCs not yet supported for Poisson solver".to_string(),
                });
            }
        }
    }

    Ok(values)
}
