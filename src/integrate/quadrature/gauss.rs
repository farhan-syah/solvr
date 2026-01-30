//! Gaussian quadrature methods.
//!
//! Gauss-Legendre quadrature is optimal for integrating polynomials.
//! An n-point rule exactly integrates polynomials of degree 2n-1.

use crate::integrate::error::{IntegrateError, IntegrateResult};

/// Gauss-Legendre quadrature rule.
///
/// Stores nodes (abscissae) and weights for Gauss-Legendre quadrature.
/// The nodes are roots of Legendre polynomials on [-1, 1].
#[derive(Debug, Clone)]
pub struct GaussLegendreQuadrature {
    /// Quadrature nodes on [-1, 1]
    pub nodes: Vec<f64>,
    /// Quadrature weights
    pub weights: Vec<f64>,
}

impl GaussLegendreQuadrature {
    /// Create a new Gauss-Legendre quadrature rule with `n` points.
    ///
    /// Uses pre-computed high-precision values for n <= 10,
    /// and the Golub-Welsch algorithm for larger n.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of quadrature points (must be >= 1)
    ///
    /// # Example
    ///
    /// ```
    /// use solvr::integrate::GaussLegendreQuadrature;
    ///
    /// let quad = GaussLegendreQuadrature::new(5).unwrap();
    /// assert_eq!(quad.nodes.len(), 5);
    /// assert_eq!(quad.weights.len(), 5);
    /// ```
    pub fn new(n: usize) -> IntegrateResult<Self> {
        if n == 0 {
            return Err(IntegrateError::InvalidParameter {
                parameter: "n".to_string(),
                message: "need at least 1 quadrature point".to_string(),
            });
        }

        match n {
            1 => Ok(Self::gauss_legendre_1()),
            2 => Ok(Self::gauss_legendre_2()),
            3 => Ok(Self::gauss_legendre_3()),
            4 => Ok(Self::gauss_legendre_4()),
            5 => Ok(Self::gauss_legendre_5()),
            6 => Ok(Self::gauss_legendre_6()),
            7 => Ok(Self::gauss_legendre_7()),
            8 => Ok(Self::gauss_legendre_8()),
            9 => Ok(Self::gauss_legendre_9()),
            10 => Ok(Self::gauss_legendre_10()),
            _ => Self::compute_nodes_weights(n),
        }
    }

    /// Integrate a function over [a, b].
    ///
    /// # Arguments
    ///
    /// * `f` - Function to integrate
    /// * `a` - Lower bound
    /// * `b` - Upper bound
    ///
    /// # Example
    ///
    /// ```
    /// use solvr::integrate::GaussLegendreQuadrature;
    ///
    /// let quad = GaussLegendreQuadrature::new(5).unwrap();
    /// let result = quad.integrate(|x| x * x, 0.0, 1.0);
    /// // Exact value is 1/3
    /// assert!((result - 1.0/3.0).abs() < 1e-10);
    /// ```
    pub fn integrate<F>(&self, f: F, a: f64, b: f64) -> f64
    where
        F: Fn(f64) -> f64,
    {
        // Transform from [-1, 1] to [a, b]
        let mid = (a + b) / 2.0;
        let half_width = (b - a) / 2.0;

        let mut result = 0.0;
        for (&node, &weight) in self.nodes.iter().zip(&self.weights) {
            let x = mid + half_width * node;
            result += weight * f(x);
        }

        result * half_width
    }

    // Pre-computed quadrature rules for n = 1 to 10
    fn gauss_legendre_1() -> Self {
        Self {
            nodes: vec![0.0],
            weights: vec![2.0],
        }
    }

    fn gauss_legendre_2() -> Self {
        let x = 0.5773502691896257_f64; // 1/sqrt(3)
        Self {
            nodes: vec![-x, x],
            weights: vec![1.0, 1.0],
        }
    }

    fn gauss_legendre_3() -> Self {
        let x = 0.7745966692414834_f64; // sqrt(3/5)
        Self {
            nodes: vec![-x, 0.0, x],
            weights: vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0],
        }
    }

    fn gauss_legendre_4() -> Self {
        Self {
            nodes: vec![
                -0.8611363115940526,
                -0.3399810435848563,
                0.3399810435848563,
                0.8611363115940526,
            ],
            weights: vec![
                0.3478548451374538,
                0.6521451548625461,
                0.6521451548625461,
                0.3478548451374538,
            ],
        }
    }

    fn gauss_legendre_5() -> Self {
        Self {
            nodes: vec![
                -0.906_179_845_938_664,
                -0.5384693101056831,
                0.0,
                0.5384693101056831,
                0.906_179_845_938_664,
            ],
            weights: vec![
                0.2369268850561891,
                0.4786286704993665,
                0.5688888888888889,
                0.4786286704993665,
                0.2369268850561891,
            ],
        }
    }

    fn gauss_legendre_6() -> Self {
        Self {
            nodes: vec![
                -0.932_469_514_203_152,
                -0.6612093864662645,
                -0.2386191860831969,
                0.2386191860831969,
                0.6612093864662645,
                0.932_469_514_203_152,
            ],
            weights: vec![
                0.1713244923791704,
                0.3607615730481386,
                0.467_913_934_572_691,
                0.467_913_934_572_691,
                0.3607615730481386,
                0.1713244923791704,
            ],
        }
    }

    fn gauss_legendre_7() -> Self {
        Self {
            nodes: vec![
                -0.9491079123427585,
                -0.7415311855993945,
                -0.4058451513773972,
                0.0,
                0.4058451513773972,
                0.7415311855993945,
                0.9491079123427585,
            ],
            weights: vec![
                0.1294849661688697,
                0.2797053914892766,
                0.3818300505051189,
                0.4179591836734694,
                0.3818300505051189,
                0.2797053914892766,
                0.1294849661688697,
            ],
        }
    }

    fn gauss_legendre_8() -> Self {
        Self {
            nodes: vec![
                -0.9602898564975363,
                -0.7966664774136267,
                -0.525_532_409_916_329,
                -0.1834346424956498,
                0.1834346424956498,
                0.525_532_409_916_329,
                0.7966664774136267,
                0.9602898564975363,
            ],
            weights: vec![
                0.1012285362903763,
                0.2223810344533745,
                0.3137066458778873,
                0.362_683_783_378_362,
                0.362_683_783_378_362,
                0.3137066458778873,
                0.2223810344533745,
                0.1012285362903763,
            ],
        }
    }

    fn gauss_legendre_9() -> Self {
        Self {
            nodes: vec![
                -0.9681602395076261,
                -0.8360311073266358,
                -0.6133714327005904,
                -0.3242534234038089,
                0.0,
                0.3242534234038089,
                0.6133714327005904,
                0.8360311073266358,
                0.9681602395076261,
            ],
            weights: vec![
                0.0812743883615744,
                0.1806481606948574,
                0.2606106964029354,
                0.3123470770400029,
                0.3302393550012598,
                0.3123470770400029,
                0.2606106964029354,
                0.1806481606948574,
                0.0812743883615744,
            ],
        }
    }

    fn gauss_legendre_10() -> Self {
        Self {
            nodes: vec![
                -0.9739065285171717,
                -0.8650633666889845,
                -0.6794095682990244,
                -0.4333953941292472,
                -0.1488743389816312,
                0.1488743389816312,
                0.4333953941292472,
                0.6794095682990244,
                0.8650633666889845,
                0.9739065285171717,
            ],
            weights: vec![
                0.0666713443086881,
                0.1494513491505806,
                0.219_086_362_515_982,
                0.2692667193099963,
                0.2955242247147529,
                0.2955242247147529,
                0.2692667193099963,
                0.219_086_362_515_982,
                0.1494513491505806,
                0.0666713443086881,
            ],
        }
    }

    /// Compute nodes and weights using Newton-Raphson on Legendre polynomials.
    ///
    /// Finds the roots of Legendre polynomial P_n(x) using Newton's method
    /// with initial guesses from Chebyshev nodes, then computes weights.
    fn compute_nodes_weights(n: usize) -> IntegrateResult<Self> {
        let mut nodes = Vec::with_capacity(n);
        let mut weights = Vec::with_capacity(n);

        let eps = 1e-15;
        let max_iter = 100;

        // We only need to find roots for the positive half due to symmetry
        let m = n.div_ceil(2);

        for i in 0..m {
            // Initial guess: Chebyshev nodes
            let mut x = ((4 * i + 3) as f64 / (4 * n + 2) as f64 * std::f64::consts::PI).cos();

            // Newton-Raphson iteration
            for _ in 0..max_iter {
                let (p, dp) = legendre_eval(n, x);

                let dx = p / dp;
                x -= dx;

                if dx.abs() < eps {
                    break;
                }
            }

            // Compute weight using the derivative
            let (_, dp) = legendre_eval(n, x);
            let w = 2.0 / ((1.0 - x * x) * dp * dp);

            // Store both the node and its symmetric counterpart
            if i != n - 1 - i {
                // Positive node
                nodes.push(x);
                weights.push(w);
                // Negative node (symmetric)
                nodes.push(-x);
                weights.push(w);
            } else {
                // Middle node (x = 0 for odd n)
                nodes.push(x);
                weights.push(w);
            }
        }

        // Sort nodes
        let mut pairs: Vec<(f64, f64)> = nodes.into_iter().zip(weights).collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let (sorted_nodes, sorted_weights): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();

        Ok(Self {
            nodes: sorted_nodes,
            weights: sorted_weights,
        })
    }
}

/// Evaluate Legendre polynomial P_n(x) and its derivative P_n'(x).
///
/// Uses the three-term recurrence relation for Legendre polynomials.
fn legendre_eval(n: usize, x: f64) -> (f64, f64) {
    if n == 0 {
        return (1.0, 0.0);
    }
    if n == 1 {
        return (x, 1.0);
    }

    let mut p_prev = 1.0; // P_0(x)
    let mut p_curr = x; // P_1(x)
    let mut dp_prev = 0.0; // P'_0(x)
    let mut dp_curr = 1.0; // P'_1(x)

    for k in 1..n {
        let k_f64 = k as f64;

        // P_{k+1}(x) = ((2k+1)*x*P_k(x) - k*P_{k-1}(x)) / (k+1)
        let p_next = ((2.0 * k_f64 + 1.0) * x * p_curr - k_f64 * p_prev) / (k_f64 + 1.0);

        // P'_{k+1}(x) = ((2k+1)*(P_k(x) + x*P'_k(x)) - k*P'_{k-1}(x)) / (k+1)
        let dp_next =
            ((2.0 * k_f64 + 1.0) * (p_curr + x * dp_curr) - k_f64 * dp_prev) / (k_f64 + 1.0);

        p_prev = p_curr;
        p_curr = p_next;
        dp_prev = dp_curr;
        dp_curr = dp_next;
    }

    (p_curr, dp_curr)
}

/// Fixed-order Gaussian quadrature.
///
/// Integrates a function using Gauss-Legendre quadrature with a fixed number of points.
/// An n-point rule is exact for polynomials of degree 2n-1.
///
/// # Arguments
///
/// * `f` - Function to integrate
/// * `a` - Lower bound
/// * `b` - Upper bound
/// * `n` - Number of quadrature points
///
/// # Example
///
/// ```
/// use solvr::integrate::fixed_quad;
///
/// // Integrate x^4 from 0 to 1 (exact = 0.2)
/// let result = fixed_quad(|x| x.powi(4), 0.0, 1.0, 5).unwrap();
/// // 5-point rule is exact for degree <= 9
/// assert!((result - 0.2).abs() < 1e-14);
/// ```
pub fn fixed_quad<F>(f: F, a: f64, b: f64, n: usize) -> IntegrateResult<f64>
where
    F: Fn(f64) -> f64,
{
    if a >= b {
        return Err(IntegrateError::InvalidInterval {
            a,
            b,
            context: "fixed_quad".to_string(),
        });
    }

    let quad = GaussLegendreQuadrature::new(n)?;
    Ok(quad.integrate(f, a, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_gauss_legendre_polynomial() {
        // n-point rule is exact for degree 2n-1

        // 2-point rule: exact for degree 3
        let quad = GaussLegendreQuadrature::new(2).unwrap();
        let result = quad.integrate(|x| x.powi(3), -1.0, 1.0);
        assert!(result.abs() < 1e-14); // Odd function, integral = 0

        // 3-point rule: exact for degree 5
        let quad = GaussLegendreQuadrature::new(3).unwrap();
        let result = quad.integrate(|x| x.powi(5), -1.0, 1.0);
        assert!(result.abs() < 1e-14);

        // 5-point rule: exact for degree 9
        let quad = GaussLegendreQuadrature::new(5).unwrap();
        let result = quad.integrate(|x| x.powi(9), -1.0, 1.0);
        assert!(result.abs() < 1e-13);
    }

    #[test]
    fn test_gauss_legendre_integration() {
        let quad = GaussLegendreQuadrature::new(10).unwrap();

        // Integrate x^2 from 0 to 1 = 1/3
        let result = quad.integrate(|x| x * x, 0.0, 1.0);
        assert!((result - 1.0 / 3.0).abs() < 1e-14);

        // Integrate sin(x) from 0 to pi = 2
        let result = quad.integrate(|x| x.sin(), 0.0, PI);
        assert!((result - 2.0).abs() < 1e-10);

        // Integrate exp(x) from 0 to 1 = e - 1
        let result = quad.integrate(|x| x.exp(), 0.0, 1.0);
        assert!((result - (std::f64::consts::E - 1.0)).abs() < 1e-12);
    }

    #[test]
    fn test_fixed_quad() {
        // Integrate x^4 from 0 to 1 = 0.2
        let result = fixed_quad(|x| x.powi(4), 0.0, 1.0, 5).unwrap();
        assert!((result - 0.2).abs() < 1e-14);

        // Integrate cos(x) from 0 to pi/2 = 1
        let result = fixed_quad(|x| x.cos(), 0.0, PI / 2.0, 10).unwrap();
        assert!((result - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_gauss_legendre_arbitrary_n() {
        // Test arbitrary n using Golub-Welsch
        let quad = GaussLegendreQuadrature::new(15).unwrap();
        assert_eq!(quad.nodes.len(), 15);
        assert_eq!(quad.weights.len(), 15);

        // Weights should sum to 2
        let weight_sum: f64 = quad.weights.iter().sum();
        assert!((weight_sum - 2.0).abs() < 1e-12);

        // Nodes should be symmetric
        for i in 0..7 {
            assert!((quad.nodes[i] + quad.nodes[14 - i]).abs() < 1e-12);
            assert!((quad.weights[i] - quad.weights[14 - i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_gauss_weights_sum() {
        // Weights should always sum to 2 (length of [-1, 1])
        for n in 1..=10 {
            let quad = GaussLegendreQuadrature::new(n).unwrap();
            let sum: f64 = quad.weights.iter().sum();
            assert!((sum - 2.0).abs() < 1e-12, "n={}, sum={}", n, sum);
        }
    }
}
