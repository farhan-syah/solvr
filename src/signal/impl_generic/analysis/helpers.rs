//! Helper functions for signal analysis algorithms.

use std::f64::consts::PI;

/// Compute FFT (simple DFT implementation).
pub fn compute_fft(data: &[f64]) -> Vec<(f64, f64)> {
    let n = data.len();
    let mut result = Vec::with_capacity(n);

    for k in 0..n {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, &x) in data.iter().enumerate() {
            let angle = -2.0 * PI * k as f64 * j as f64 / n as f64;
            re += x * angle.cos();
            im += x * angle.sin();
        }
        result.push((re, im));
    }

    result
}

/// Compute IFFT.
pub fn compute_ifft(data: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = data.len();
    let mut result = Vec::with_capacity(n);

    for k in 0..n {
        let mut re = 0.0;
        let mut im = 0.0;
        for (j, &(x_re, x_im)) in data.iter().enumerate() {
            let angle = 2.0 * PI * k as f64 * j as f64 / n as f64;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            re += x_re * cos_a - x_im * sin_a;
            im += x_re * sin_a + x_im * cos_a;
        }
        result.push((re / n as f64, im / n as f64));
    }

    result
}

/// Apply Butterworth lowpass filter.
pub fn apply_butter_lowpass(data: &[f64], cutoff: f64, order: usize, zero_phase: bool) -> Vec<f64> {
    // Simple first-order IIR approximation for each cascade stage
    let alpha = 2.0 * PI * cutoff / (2.0 + 2.0 * PI * cutoff);

    let mut filtered = data.to_vec();

    // Apply multiple passes for higher order
    for _ in 0..order {
        // Forward pass
        let mut y_prev = filtered[0];
        for i in 0..filtered.len() {
            let y = alpha * filtered[i] + (1.0 - alpha) * y_prev;
            filtered[i] = y;
            y_prev = y;
        }

        if zero_phase {
            // Backward pass
            y_prev = filtered[filtered.len() - 1];
            for i in (0..filtered.len()).rev() {
                let y = alpha * filtered[i] + (1.0 - alpha) * y_prev;
                filtered[i] = y;
                y_prev = y;
            }
        }
    }

    filtered
}

/// Apply FIR lowpass filter.
pub fn apply_fir_lowpass(data: &[f64], cutoff: f64, filter_len: usize) -> Vec<f64> {
    // Design sinc filter
    let half = filter_len / 2;
    let mut h = Vec::with_capacity(filter_len);

    for i in 0..filter_len {
        let n = i as f64 - half as f64;
        let sinc = if n.abs() < 1e-10 {
            2.0 * cutoff
        } else {
            (2.0 * PI * cutoff * n).sin() / (PI * n)
        };
        // Apply Hamming window
        let window = 0.54 - 0.46 * (2.0 * PI * i as f64 / (filter_len - 1) as f64).cos();
        h.push(sinc * window);
    }

    // Normalize
    let sum: f64 = h.iter().sum();
    if sum.abs() > 1e-10 {
        for c in &mut h {
            *c /= sum;
        }
    }

    // Convolve
    let n = data.len();
    let mut result = vec![0.0; n];

    for i in 0..n {
        for (j, &hj) in h.iter().enumerate() {
            let k = i as isize + j as isize - half as isize;
            if k >= 0 && (k as usize) < n {
                result[i] += data[k as usize] * hj;
            }
        }
    }

    result
}

/// Filter peaks by minimum distance.
pub fn filter_by_distance(peaks: &[usize], data: &[f64], min_distance: usize) -> Vec<usize> {
    if peaks.is_empty() {
        return vec![];
    }

    // Sort peaks by height (descending)
    let mut sorted: Vec<(usize, f64)> = peaks.iter().map(|&i| (i, data[i])).collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = vec![true; peaks.len()];
    let mut result = Vec::new();

    for (i, &(idx, _)) in sorted.iter().enumerate() {
        if keep[i] {
            result.push(idx);
            // Mark nearby peaks as removed
            for (j, &(other_idx, _)) in sorted.iter().enumerate() {
                if j != i && keep[j] {
                    let dist = if idx > other_idx {
                        idx - other_idx
                    } else {
                        other_idx - idx
                    };
                    if dist < min_distance {
                        keep[j] = false;
                    }
                }
            }
        }
    }

    result.sort();
    result
}

/// Compute peak prominences.
pub fn compute_prominences(peaks: &[usize], data: &[f64]) -> Vec<f64> {
    let n = data.len();
    let mut prominences = Vec::with_capacity(peaks.len());

    for &peak in peaks {
        let peak_height = data[peak];

        // Search left for the lowest point before reaching a higher peak
        let mut left_min = peak_height;
        for i in (0..peak).rev() {
            if data[i] > peak_height {
                break;
            }
            left_min = left_min.min(data[i]);
        }

        // Search right for the lowest point before reaching a higher peak
        let mut right_min = peak_height;
        for i in peak + 1..n {
            if data[i] > peak_height {
                break;
            }
            right_min = right_min.min(data[i]);
        }

        // Prominence is peak height minus the higher of the two bases
        let base = left_min.max(right_min);
        prominences.push(peak_height - base);
    }

    prominences
}

/// Compute Savitzky-Golay filter coefficients.
pub fn compute_savgol_coeffs(window_length: usize, polyorder: usize, deriv: usize) -> Vec<f64> {
    let half = window_length / 2;

    // Build Vandermonde matrix
    let m = polyorder + 1;
    let mut a = vec![vec![0.0; m]; window_length];

    for i in 0..window_length {
        let x = i as f64 - half as f64;
        let mut xi = 1.0;
        for j in 0..m {
            a[i][j] = xi;
            xi *= x;
        }
    }

    // Solve least squares: coeffs = (A^T A)^{-1} A^T * e_deriv
    // where e_deriv is [0, 0, ..., deriv!, 0, ...] (1 at position deriv)

    // Compute A^T A
    let mut ata = vec![vec![0.0; m]; m];
    for i in 0..m {
        for j in 0..m {
            for k in 0..window_length {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
    }

    // Compute A^T e_deriv (just the deriv-th column of A^T)
    let mut at_e = vec![0.0; m];
    let factorial: f64 = (1..=deriv).map(|x| x as f64).product();
    for k in 0..window_length {
        at_e[deriv] += a[k][deriv];
    }
    at_e[deriv] = factorial;

    // Solve (A^T A) x = A^T e_deriv using Gauss-Jordan elimination
    let mut augmented = vec![vec![0.0; m + 1]; m];
    for i in 0..m {
        for j in 0..m {
            augmented[i][j] = ata[i][j];
        }
        augmented[i][m] = at_e[i];
    }

    // Forward elimination
    for i in 0..m {
        // Find pivot
        let mut max_row = i;
        for k in i + 1..m {
            if augmented[k][i].abs() > augmented[max_row][i].abs() {
                max_row = k;
            }
        }
        augmented.swap(i, max_row);

        let pivot = augmented[i][i];
        if pivot.abs() < 1e-30 {
            continue;
        }

        for j in i..=m {
            augmented[i][j] /= pivot;
        }

        for k in 0..m {
            if k != i {
                let factor = augmented[k][i];
                for j in i..=m {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }

    let x: Vec<f64> = augmented.iter().map(|row| row[m]).collect();

    // Compute filter coefficients: h[i] = Σ x[j] * a[i][j]
    let mut coeffs = vec![0.0; window_length];
    for i in 0..window_length {
        for j in 0..m {
            coeffs[i] += x[j] * a[i][j];
        }
    }

    coeffs
}
