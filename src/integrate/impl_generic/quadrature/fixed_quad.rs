//! Fixed-order Gaussian quadrature.

use numr::error::{Error, Result};
use numr::ops::TensorOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Fixed-order Gaussian quadrature.
///
/// Integrates a function from a to b using n-point Gauss-Legendre quadrature.
/// All n evaluation points are computed in a single batch.
pub fn fixed_quad_impl<R, C, F>(client: &C, f: F, a: f64, b: f64, n: usize) -> Result<Tensor<R>>
where
    R: Runtime,
    C: TensorOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>) -> Result<Tensor<R>>,
{
    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "n",
            reason: "fixed_quad: n must be at least 1".to_string(),
        });
    }

    // Get Gauss-Legendre nodes and weights for [-1, 1]
    let (nodes, weights) = gauss_legendre_nodes_weights(n);

    // Transform nodes from [-1, 1] to [a, b]
    let half_width = (b - a) / 2.0;
    let center = (a + b) / 2.0;

    let transformed_nodes: Vec<f64> = nodes.iter().map(|&x| center + half_width * x).collect();

    // Evaluate function at all nodes in a single batch
    let x_tensor = Tensor::<R>::from_slice(&transformed_nodes, &[n], client.device());
    let f_values = f(&x_tensor)?;
    let f_data: Vec<f64> = f_values.to_vec();

    // Compute weighted sum
    let mut integral = 0.0;
    for i in 0..n {
        integral += weights[i] * f_data[i];
    }
    integral *= half_width;

    Ok(Tensor::<R>::from_slice(&[integral], &[], client.device()))
}

/// Compute Gauss-Legendre nodes and weights.
///
/// Uses Newton iteration to find roots of Legendre polynomials.
fn gauss_legendre_nodes_weights(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];

    let m = n.div_ceil(2);

    for i in 0..m {
        // Initial guess using Chebyshev approximation
        let mut z = ((i as f64 + 0.75) / (n as f64 + 0.5) * std::f64::consts::PI).cos();

        // Newton iteration to find root of Legendre polynomial
        loop {
            let (p, dp) = legendre_p_and_dp(n, z);
            let z_new = z - p / dp;

            if (z_new - z).abs() < 1e-15 {
                z = z_new;
                break;
            }
            z = z_new;
        }

        let (_, dp) = legendre_p_and_dp(n, z);
        let w = 2.0 / ((1.0 - z * z) * dp * dp);

        nodes[i] = -z;
        nodes[n - 1 - i] = z;
        weights[i] = w;
        weights[n - 1 - i] = w;
    }

    (nodes, weights)
}

/// Evaluate Legendre polynomial P_n(x) and its derivative.
fn legendre_p_and_dp(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }

    let mut p_prev = 1.0;
    let mut p_curr = x;

    for k in 2..=n {
        let p_next = ((2 * k - 1) as f64 * x * p_curr - (k - 1) as f64 * p_prev) / k as f64;
        p_prev = p_curr;
        p_curr = p_next;
    }

    // Derivative: P'_n(x) = n * (x * P_n - P_{n-1}) / (x^2 - 1)
    let dp = n as f64 * (x * p_curr - p_prev) / (x * x - 1.0);

    (p_curr, dp)
}
