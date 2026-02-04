//! Peak detection in signals.

use super::helpers::{compute_prominences, filter_by_distance};
use crate::signal::traits::analysis::{PeakParams, PeakResult};
use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Find peaks in a signal.
pub fn find_peaks_impl<R, C>(
    _client: &C,
    x: &Tensor<R>,
    params: PeakParams,
) -> Result<PeakResult<R>>
where
    R: Runtime,
    C: ScalarOps<R> + TensorOps<R> + RuntimeClient<R>,
{
    let x_data: Vec<f64> = x.to_vec();
    let n = x_data.len();
    let device = x.device();

    if n < 3 {
        // Need at least 3 points to find a peak
        return Ok(PeakResult {
            indices: vec![],
            heights: Tensor::from_slice(&[] as &[f64], &[0], device),
            prominences: None,
        });
    }

    // Find all local maxima
    let mut peaks: Vec<usize> = Vec::new();

    for i in 1..n - 1 {
        if x_data[i] > x_data[i - 1] && x_data[i] > x_data[i + 1] {
            peaks.push(i);
        }
    }

    // Filter by height
    if let Some(min_height) = params.height {
        peaks.retain(|&i| x_data[i] >= min_height);
    }

    // Filter by threshold (difference from neighbors)
    if let Some(threshold) = params.threshold {
        peaks.retain(|&i| {
            let left_diff = x_data[i] - x_data[i - 1];
            let right_diff = x_data[i] - x_data[i + 1];
            left_diff >= threshold && right_diff >= threshold
        });
    }

    // Filter by distance (keep highest peak when peaks are too close)
    if let Some(min_distance) = params.distance {
        if min_distance > 0 {
            peaks = filter_by_distance(&peaks, &x_data, min_distance);
        }
    }

    // Compute prominences if requested
    let prominences = if params.prominence.is_some() {
        let proms = compute_prominences(&peaks, &x_data);

        // Filter by prominence
        if let Some(min_prom) = params.prominence {
            let filtered: Vec<(usize, f64)> = peaks
                .iter()
                .zip(proms.iter())
                .filter(|(_, p)| **p >= min_prom)
                .map(|(i, p)| (*i, *p))
                .collect();

            peaks = filtered.iter().map(|(i, _)| *i).collect();
            let new_proms: Vec<f64> = filtered.iter().map(|(_, p)| *p).collect();
            Some(Tensor::from_slice(&new_proms, &[new_proms.len()], device))
        } else {
            Some(Tensor::from_slice(&proms, &[proms.len()], device))
        }
    } else {
        None
    };

    // Extract heights
    let heights: Vec<f64> = peaks.iter().map(|&i| x_data[i]).collect();

    Ok(PeakResult {
        indices: peaks,
        heights: Tensor::from_slice(&heights, &[heights.len()], device),
        prominences,
    })
}
