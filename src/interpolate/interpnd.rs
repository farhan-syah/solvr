//! N-dimensional interpolation on regular grids.
//!
//! This module provides interpolation over N-dimensional rectilinear grids,
//! where each dimension has its own 1D coordinate array.
//!
//! # Example
//!
//! ```ignore
//! use solvr::interpolate::{RegularGridInterpolator, InterpNdMethod};
//!
//! // Create a 2D grid: x = [0, 1, 2], y = [0, 1]
//! let x = Tensor::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
//! let y = Tensor::from_slice(&[0.0, 1.0], &[2], &device);
//! // Values on 2x3 grid (shape matches [len(y), len(x)])
//! let z = Tensor::from_slice(&[0.0, 1.0, 2.0, 1.0, 2.0, 3.0], &[2, 3], &device);
//!
//! let interp = RegularGridInterpolator::new(&client, &[&x, &y], &z, InterpNdMethod::Linear)?;
//! let result = interp.evaluate(&client, &query_points)?;
//! ```

use crate::interpolate::error::{InterpolateError, InterpolateResult};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Interpolation method for N-dimensional grids.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterpNdMethod {
    /// Nearest neighbor interpolation.
    Nearest,
    /// Multilinear interpolation (default).
    #[default]
    Linear,
}

/// Behavior when query points are outside the grid domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExtrapolateMode {
    /// Return an error for out-of-bounds queries.
    #[default]
    Error,
    /// Return NaN for out-of-bounds queries.
    Nan,
    /// Extrapolate beyond grid bounds (use boundary values for nearest).
    Extrapolate,
}

/// N-dimensional interpolation on a regular (rectilinear) grid.
///
/// Supports any number of dimensions. For a grid with dimensions [d0, d1, ..., dn-1],
/// you provide n coordinate arrays where the i-th array has length di, and a values
/// tensor with shape [d0, d1, ..., dn-1].
pub struct RegularGridInterpolator<R: Runtime> {
    /// Coordinate arrays for each dimension.
    points: Vec<Vec<f64>>,
    /// Grid values (N-dimensional).
    values: Vec<f64>,
    /// Shape of the values array.
    shape: Vec<usize>,
    /// Number of dimensions.
    n_dims: usize,
    /// Interpolation method.
    method: InterpNdMethod,
    /// How to handle out-of-bounds queries.
    extrapolate: ExtrapolateMode,
    /// Device for output tensors.
    device: R::Device,
}

impl<R: Runtime> RegularGridInterpolator<R> {
    /// Create a new N-dimensional grid interpolator.
    ///
    /// # Arguments
    ///
    /// * `client` - Runtime client
    /// * `points` - Slice of 1D tensors, one per dimension. Each must be strictly increasing.
    /// * `values` - N-dimensional tensor of values. Shape must match [len(points[0]), len(points[1]), ...]
    /// * `method` - Interpolation method (Nearest or Linear)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Any coordinate array has fewer than 2 points
    /// - Coordinate arrays are not strictly increasing
    /// - Values shape doesn't match coordinate array lengths
    pub fn new<C: RuntimeClient<R>>(
        _client: &C,
        points: &[&Tensor<R>],
        values: &Tensor<R>,
        method: InterpNdMethod,
    ) -> InterpolateResult<Self> {
        Self::with_extrapolate(_client, points, values, method, ExtrapolateMode::Error)
    }

    /// Create a new interpolator with custom extrapolation behavior.
    pub fn with_extrapolate<C: RuntimeClient<R>>(
        _client: &C,
        points: &[&Tensor<R>],
        values: &Tensor<R>,
        method: InterpNdMethod,
        extrapolate: ExtrapolateMode,
    ) -> InterpolateResult<Self> {
        let ndim = points.len();
        if ndim == 0 {
            return Err(InterpolateError::InvalidParameter {
                parameter: "points".to_string(),
                message: "At least one dimension required".to_string(),
            });
        }

        let values_shape = values.shape();
        if values_shape.len() != ndim {
            return Err(InterpolateError::DimensionMismatch {
                expected: ndim,
                actual: values_shape.len(),
                context: "RegularGridInterpolator::new (values dimensions)".to_string(),
            });
        }

        let mut point_vecs = Vec::with_capacity(ndim);
        let mut shape = Vec::with_capacity(ndim);

        for (dim, &pts) in points.iter().enumerate() {
            let pts_shape = pts.shape();
            if pts_shape.len() != 1 {
                return Err(InterpolateError::InvalidParameter {
                    parameter: format!("points[{}]", dim),
                    message: "Coordinate arrays must be 1D".to_string(),
                });
            }

            let n = pts_shape[0];
            if n < 2 {
                return Err(InterpolateError::InsufficientData {
                    required: 2,
                    actual: n,
                    context: format!("RegularGridInterpolator dimension {}", dim),
                });
            }

            if n != values_shape[dim] {
                return Err(InterpolateError::ShapeMismatch {
                    expected: n,
                    actual: values_shape[dim],
                    context: format!(
                        "RegularGridInterpolator dimension {} (points vs values)",
                        dim
                    ),
                });
            }

            let pts_data: Vec<f64> = pts.to_vec();

            // Check strictly increasing
            for i in 1..n {
                if pts_data[i] <= pts_data[i - 1] {
                    return Err(InterpolateError::NotMonotonic {
                        context: format!("RegularGridInterpolator dimension {}", dim),
                    });
                }
            }

            shape.push(n);
            point_vecs.push(pts_data);
        }

        let values_data: Vec<f64> = values.to_vec();
        let device = values.device().clone();

        Ok(Self {
            points: point_vecs,
            values: values_data,
            shape,
            n_dims: ndim,
            method,
            extrapolate,
            device,
        })
    }

    /// Evaluate the interpolant at query points.
    ///
    /// # Arguments
    ///
    /// * `client` - Runtime client
    /// * `xi` - Query points as 2D tensor of shape [n_points, ndim]
    ///
    /// # Returns
    ///
    /// 1D tensor of interpolated values with length n_points.
    pub fn evaluate<C: RuntimeClient<R>>(
        &self,
        _client: &C,
        xi: &Tensor<R>,
    ) -> InterpolateResult<Tensor<R>> {
        let xi_shape = xi.shape();
        if xi_shape.len() != 2 {
            return Err(InterpolateError::InvalidParameter {
                parameter: "xi".to_string(),
                message: format!(
                    "Query points must be 2D [n_points, ndim], got {:?}",
                    xi_shape
                ),
            });
        }

        let n_points = xi_shape[0];
        let query_ndim = xi_shape[1];

        if query_ndim != self.n_dims {
            return Err(InterpolateError::DimensionMismatch {
                expected: self.n_dims,
                actual: query_ndim,
                context: "RegularGridInterpolator::evaluate (query dimensions)".to_string(),
            });
        }

        let xi_data: Vec<f64> = xi.to_vec();
        let mut results = Vec::with_capacity(n_points);

        for i in 0..n_points {
            let point: Vec<f64> = (0..self.n_dims)
                .map(|d| xi_data[i * self.n_dims + d])
                .collect();

            let value = self.evaluate_single(&point)?;
            results.push(value);
        }

        Ok(Tensor::from_slice(&results, &[n_points], &self.device))
    }

    /// Evaluate at a single point.
    fn evaluate_single(&self, point: &[f64]) -> InterpolateResult<f64> {
        // Find intervals and check bounds for each dimension
        let mut indices = Vec::with_capacity(self.n_dims);
        let mut fracs = Vec::with_capacity(self.n_dims);

        for (d, (&x, pts)) in point.iter().zip(self.points.iter()).enumerate() {
            let n = pts.len();

            // Check bounds
            if x < pts[0] || x > pts[n - 1] {
                match self.extrapolate {
                    ExtrapolateMode::Error => {
                        return Err(InterpolateError::OutOfDomainNd {
                            dimension: d,
                            point: x,
                            min: pts[0],
                            max: pts[n - 1],
                            context: "RegularGridInterpolator::evaluate".to_string(),
                        });
                    }
                    ExtrapolateMode::Nan => {
                        return Ok(f64::NAN);
                    }
                    ExtrapolateMode::Extrapolate => {
                        // Clamp to boundary
                        let clamped = x.clamp(pts[0], pts[n - 1]);
                        let (idx, frac) = self.find_interval(pts, clamped);
                        indices.push(idx);
                        fracs.push(frac);
                        continue;
                    }
                }
            }

            let (idx, frac) = self.find_interval(pts, x);
            indices.push(idx);
            fracs.push(frac);
        }

        match self.method {
            InterpNdMethod::Nearest => self.interp_nearest(&indices, &fracs),
            InterpNdMethod::Linear => self.interp_linear(&indices, &fracs),
        }
    }

    /// Find interval index and fractional position for a value.
    fn find_interval(&self, pts: &[f64], x: f64) -> (usize, f64) {
        let n = pts.len();

        // Binary search for interval
        let mut lo = 0;
        let mut hi = n - 1;

        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if pts[mid] <= x {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        // Handle edge case: x exactly at upper bound
        if lo == n - 2 && x == pts[n - 1] {
            return (lo, 1.0);
        }

        let frac = (x - pts[lo]) / (pts[lo + 1] - pts[lo]);
        (lo, frac)
    }

    /// Nearest neighbor interpolation.
    fn interp_nearest(&self, indices: &[usize], fracs: &[f64]) -> InterpolateResult<f64> {
        // Round to nearest grid point
        let mut idx = Vec::with_capacity(self.n_dims);
        for d in 0..self.n_dims {
            let i = if fracs[d] < 0.5 {
                indices[d]
            } else {
                (indices[d] + 1).min(self.shape[d] - 1)
            };
            idx.push(i);
        }

        Ok(self.get_value(&idx))
    }

    /// Multilinear interpolation.
    fn interp_linear(&self, indices: &[usize], fracs: &[f64]) -> InterpolateResult<f64> {
        // Number of vertices in the hypercube: 2^ndim
        let n_vertices = 1 << self.n_dims;
        let mut result = 0.0;

        for vertex in 0..n_vertices {
            // Build index for this vertex and compute weight
            let mut idx = Vec::with_capacity(self.n_dims);
            let mut weight = 1.0;

            for d in 0..self.n_dims {
                let use_upper = (vertex >> d) & 1 == 1;
                if use_upper {
                    idx.push((indices[d] + 1).min(self.shape[d] - 1));
                    weight *= fracs[d];
                } else {
                    idx.push(indices[d]);
                    weight *= 1.0 - fracs[d];
                }
            }

            result += weight * self.get_value(&idx);
        }

        Ok(result)
    }

    /// Get value at a multi-dimensional index (row-major order).
    fn get_value(&self, idx: &[usize]) -> f64 {
        let mut flat_idx = 0;
        let mut stride = 1;

        // Row-major: last dimension varies fastest
        for d in (0..self.n_dims).rev() {
            flat_idx += idx[d] * stride;
            stride *= self.shape[d];
        }

        self.values[flat_idx]
    }

    /// Returns the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.n_dims
    }

    /// Returns the shape of the grid.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the domain bounds as [(min, max), ...] for each dimension.
    pub fn bounds(&self) -> Vec<(f64, f64)> {
        self.points
            .iter()
            .map(|pts| (pts[0], pts[pts.len() - 1]))
            .collect()
    }

    /// Returns the interpolation method.
    pub fn method(&self) -> InterpNdMethod {
        self.method
    }

    /// Returns the extrapolation mode.
    pub fn extrapolate_mode(&self) -> ExtrapolateMode {
        self.extrapolate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_1d_linear() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 4.0], &[3], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&x], &y, InterpNdMethod::Linear).unwrap();

        // Query at midpoints
        let xi = Tensor::<CpuRuntime>::from_slice(&[0.5, 1.5], &[2, 1], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!((result_data[0] - 1.0).abs() < 1e-10);
        assert!((result_data[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_2d_linear_simple() {
        let (device, client) = setup();

        // 2x3 grid
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        // z[i,j] = i + j (row-major: shape [2, 3])
        let z = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 1.0, 2.0, 3.0], &[2, 3], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&y, &x], &z, InterpNdMethod::Linear).unwrap();

        // Query at grid points
        let xi = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 2.0], &[2, 2], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!((result_data[0] - 0.0).abs() < 1e-10); // z[0,0]
        assert!((result_data[1] - 3.0).abs() < 1e-10); // z[1,2]
    }

    #[test]
    fn test_2d_linear_interpolation() {
        let (device, client) = setup();

        // Unit square with values at corners
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        // z = x + y at corners: [0,0]=0, [0,1]=1, [1,0]=1, [1,1]=2
        let z = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 1.0, 2.0], &[2, 2], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&y, &x], &z, InterpNdMethod::Linear).unwrap();

        // Query at center
        let xi = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5], &[1, 2], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Bilinear interpolation at center should give 1.0
        assert!((result_data[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_nearest_neighbor() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[10.0, 20.0, 30.0], &[3], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&x], &y, InterpNdMethod::Nearest).unwrap();

        // Query closer to different points
        let xi = Tensor::<CpuRuntime>::from_slice(&[0.3, 0.7, 1.4], &[3, 1], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!((result_data[0] - 10.0).abs() < 1e-10); // Closer to 0
        assert!((result_data[1] - 20.0).abs() < 1e-10); // Closer to 1
        assert!((result_data[2] - 20.0).abs() < 1e-10); // Closer to 1
    }

    #[test]
    fn test_out_of_bounds_error() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&x], &y, InterpNdMethod::Linear).unwrap();

        let xi = Tensor::<CpuRuntime>::from_slice(&[-0.5], &[1, 1], &device);
        let result = interp.evaluate(&client, &xi);

        assert!(matches!(
            result,
            Err(InterpolateError::OutOfDomainNd { .. })
        ));
    }

    #[test]
    fn test_extrapolate_nan() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);

        let interp = RegularGridInterpolator::with_extrapolate(
            &client,
            &[&x],
            &y,
            InterpNdMethod::Linear,
            ExtrapolateMode::Nan,
        )
        .unwrap();

        let xi = Tensor::<CpuRuntime>::from_slice(&[-0.5], &[1, 1], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!(result_data[0].is_nan());
    }

    #[test]
    fn test_extrapolate_clamp() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[10.0, 20.0], &[2], &device);

        let interp = RegularGridInterpolator::with_extrapolate(
            &client,
            &[&x],
            &y,
            InterpNdMethod::Linear,
            ExtrapolateMode::Extrapolate,
        )
        .unwrap();

        // Query before and after bounds
        let xi = Tensor::<CpuRuntime>::from_slice(&[-1.0, 2.0], &[2, 1], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        // Should clamp to boundary values
        assert!((result_data[0] - 10.0).abs() < 1e-10); // Clamped to x=0
        assert!((result_data[1] - 20.0).abs() < 1e-10); // Clamped to x=1
    }

    #[test]
    fn test_dimension_validation() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        // Wrong shape: 2D values for 1D grid
        let z = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0, 3.0], &[2, 2], &device);

        let result = RegularGridInterpolator::new(&client, &[&x], &z, InterpNdMethod::Linear);
        assert!(matches!(
            result,
            Err(InterpolateError::DimensionMismatch { .. })
        ));

        // Correct 2D
        let result =
            RegularGridInterpolator::new(&client, &[&x, &y], &z, InterpNdMethod::Linear).unwrap();
        assert_eq!(result.ndim(), 2);
    }

    #[test]
    fn test_shape_mismatch() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);

        let result = RegularGridInterpolator::new(&client, &[&x], &y, InterpNdMethod::Linear);
        assert!(matches!(
            result,
            Err(InterpolateError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_not_monotonic() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 2.0, 1.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0, 2.0], &[3], &device);

        let result = RegularGridInterpolator::new(&client, &[&x], &y, InterpNdMethod::Linear);
        assert!(matches!(result, Err(InterpolateError::NotMonotonic { .. })));
    }

    #[test]
    fn test_3d_interpolation() {
        let (device, client) = setup();

        // 2x2x2 grid
        let x = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);
        let z = Tensor::<CpuRuntime>::from_slice(&[0.0, 1.0], &[2], &device);

        // f(x,y,z) = x + y + z at corners
        let values = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 3.0],
            &[2, 2, 2],
            &device,
        );

        let interp =
            RegularGridInterpolator::new(&client, &[&x, &y, &z], &values, InterpNdMethod::Linear)
                .unwrap();

        // Query at center: should be 1.5 (average of all corners)
        let xi = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5, 0.5], &[1, 3], &device);
        let result = interp.evaluate(&client, &xi).unwrap();
        let result_data: Vec<f64> = result.to_vec();

        assert!((result_data[0] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_bounds() {
        let (device, client) = setup();

        let x = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0], &[3], &device);
        let y = Tensor::<CpuRuntime>::from_slice(&[-1.0, 0.0], &[2], &device);
        let z = Tensor::<CpuRuntime>::from_slice(&[0.0; 6], &[2, 3], &device);

        let interp =
            RegularGridInterpolator::new(&client, &[&y, &x], &z, InterpNdMethod::Linear).unwrap();

        let bounds = interp.bounds();
        assert_eq!(bounds.len(), 2);
        assert!((bounds[0].0 - (-1.0)).abs() < 1e-10);
        assert!((bounds[0].1 - 0.0).abs() < 1e-10);
        assert!((bounds[1].0 - 1.0).abs() < 1e-10);
        assert!((bounds[1].1 - 3.0).abs() < 1e-10);
    }
}
