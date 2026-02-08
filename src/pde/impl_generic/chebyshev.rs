//! Shared Chebyshev spectral utilities.

use std::f64::consts::PI;

/// Build the Chebyshev differentiation matrix D for N+1 points.
///
/// Returns a flat Vec of length (N+1)^2 in row-major order.
/// Chebyshev nodes: x_j = cos(j*pi/N), j = 0..N.
pub fn chebyshev_diff_matrix(n: usize) -> Vec<f64> {
    let np1 = n + 1;
    let mut d = vec![0.0; np1 * np1];

    let x: Vec<f64> = (0..np1).map(|j| (j as f64 * PI / n as f64).cos()).collect();

    // Barycentric weights
    let mut c = vec![1.0; np1];
    c[0] = 2.0;
    c[n] = 2.0;

    for i in 0..np1 {
        for j in 0..np1 {
            if i != j {
                let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
                d[i * np1 + j] = (c[i] / c[j]) * sign / (x[i] - x[j]);
            }
        }
    }

    // Diagonal entries: D[i,i] = -sum_{j!=i} D[i,j]
    for i in 0..np1 {
        let mut sum = 0.0;
        for j in 0..np1 {
            if j != i {
                sum += d[i * np1 + j];
            }
        }
        d[i * np1 + i] = -sum;
    }

    d
}

/// Build Chebyshev nodes: x_j = cos(j*pi/N), j = 0..N.
pub fn chebyshev_nodes(n: usize) -> Vec<f64> {
    let np1 = n + 1;
    (0..np1).map(|j| (j as f64 * PI / n as f64).cos()).collect()
}
