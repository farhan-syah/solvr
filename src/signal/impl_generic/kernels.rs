//! Filter kernel generation for N-D filters and edge detection.
//!
//! Generates Gaussian, Sobel, Prewitt, Laplace, and uniform kernels
//! as on-device tensors using numr ops. All kernel generation stays on device
//! with no CPU roundtrips.

use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, UnaryOps, UtilityOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generate a 1D Gaussian kernel.
///
/// Generates the Gaussian function G(x) = exp(-x^2 / (2*sigma^2))
/// or its derivatives depending on the order parameter.
///
/// # Arguments
///
/// * `client` - Runtime client
/// * `sigma` - Standard deviation (must be positive)
/// * `order` - Derivative order (0 = Gaussian, 1 = first derivative, 2 = second derivative, 3 = third)
/// * `truncate` - Truncate at this many standard deviations (default 4.0)
/// * `dtype` - Output dtype (F32 or F64)
///
/// # Returns
///
/// 1D Gaussian kernel tensor with size = 2*radius + 1
///
/// # Example
///
/// ```ignore
/// // Generate a Gaussian smoothing kernel with sigma=1.0, truncate at 4 standard deviations
/// let kernel = gaussian_kernel_1d(&client, 1.0, 0, 4.0, DType::F64)?;
/// ```
pub fn gaussian_kernel_1d<R, C>(
    client: &C,
    sigma: f64,
    order: usize,
    truncate: f64,
    dtype: DType,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + UnaryOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    if sigma <= 0.0 {
        return Err(Error::InvalidArgument {
            arg: "sigma",
            reason: "Gaussian sigma must be positive".to_string(),
        });
    }

    if order > 3 {
        return Err(Error::InvalidArgument {
            arg: "order",
            reason: "Gaussian derivative order must be 0-3".to_string(),
        });
    }

    let radius = (truncate * sigma + 0.5) as usize;
    let size = 2 * radius + 1;

    if size == 0 {
        return Err(Error::InvalidArgument {
            arg: "sigma",
            reason: "Gaussian kernel size is zero (sigma too small)".to_string(),
        });
    }

    // Generate positions: -radius to +radius
    // arange(0..size) produces [0, 1, 2, ..., size-1]
    let positions = client.arange(0.0, size as f64, 1.0, dtype)?;
    // Subtract radius to get [-radius, -radius+1, ..., radius]
    let positions = client.add_scalar(&positions, -(radius as f64))?;

    // Compute Gaussian: exp(-x^2 / (2*sigma^2))
    let x2 = client.mul(&positions, &positions)?;
    let neg_half_inv_sigma2 = -0.5 / (sigma * sigma);
    let scaled = client.mul_scalar(&x2, neg_half_inv_sigma2)?;
    let gaussian = client.exp(&scaled)?;

    match order {
        0 => {
            // Normalize so sum = 1 (standard Gaussian)
            let sum = client.sum(&gaussian, &[0], false)?;
            client.div(&gaussian, &sum)
        }
        1 => {
            // First derivative: -x / sigma^2 * G(x)
            let inv_sigma2 = -1.0 / (sigma * sigma);
            let factor = client.mul_scalar(&positions, inv_sigma2)?;
            let deriv = client.mul(&factor, &gaussian)?;
            // Normalize: scipy normalizes derivative such that sum of x*kernel ≈ -1
            let weighted = client.mul(&positions, &deriv)?;
            let weight_sum = client.sum(&weighted, &[0], false)?;
            // To normalize: divide by -sum (to make integral of x*G'(x) = -1)
            let neg_sum = client.mul_scalar(&weight_sum, -1.0)?;
            client.div(&deriv, &neg_sum)
        }
        2 => {
            // Second derivative: (x^2/sigma^4 - 1/sigma^2) * G(x)
            let inv_sigma2 = 1.0 / (sigma * sigma);
            let inv_sigma4 = inv_sigma2 * inv_sigma2;
            let term1 = client.mul_scalar(&x2, inv_sigma4)?;
            let term2 = client.add_scalar(&term1, -inv_sigma2)?;
            let deriv = client.mul(&term2, &gaussian)?;
            // Normalize so sum of x^2*kernel = 1
            let x2_weighted = client.mul(&x2, &deriv)?;
            let x2_sum = client.sum(&x2_weighted, &[0], false)?;
            client.div(&deriv, &x2_sum)
        }
        _ => {
            // Order 3: third derivative
            // (-x^3/sigma^6 + 3x/sigma^4) * G(x)
            let inv_sigma2 = 1.0 / (sigma * sigma);
            let inv_sigma4 = inv_sigma2 * inv_sigma2;
            let inv_sigma6 = inv_sigma4 * inv_sigma2;

            // Compute x^3 = x^2 * x
            let x3 = client.mul(&x2, &positions)?;

            // First term: -x^3 / sigma^6
            let t1 = client.mul_scalar(&x3, -inv_sigma6)?;
            // Second term: 3*x / sigma^4
            let t2 = client.mul_scalar(&positions, 3.0 * inv_sigma4)?;
            // Sum terms
            let factor = client.add(&t1, &t2)?;
            // Multiply by Gaussian
            let deriv = client.mul(&factor, &gaussian)?;
            Ok(deriv)
        }
    }
}

/// Generate a 1D uniform (box) kernel.
///
/// Creates a kernel of given size where all values are 1/size,
/// implementing a simple moving average / box filter.
///
/// # Arguments
///
/// * `client` - Runtime client
/// * `size` - Kernel size (must be positive)
/// * `dtype` - Output dtype
///
/// # Returns
///
/// 1D uniform kernel tensor with all elements equal to 1/size
pub fn uniform_kernel_1d<R, C>(client: &C, size: usize, dtype: DType) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    if size == 0 {
        return Err(Error::InvalidArgument {
            arg: "size",
            reason: "Uniform kernel size must be positive".to_string(),
        });
    }

    let value = 1.0 / size as f64;
    client.fill(&[size], value, dtype)
}

/// Generate a 1D derivative or smoothing kernel for edge detection (Sobel/Prewitt).
///
/// Used for separable Sobel and Prewitt filtering. One kernel applies the derivative
/// ([-1, 0, 1] or normalized variant), the other applies smoothing.
///
/// # Arguments
///
/// * `client` - Runtime client for tensor creation
/// * `kind` - "sobel" uses [1, 2, 1] smoothing, "prewitt" uses [1, 1, 1]
/// * `derivative` - If true, returns derivative kernel [-1, 0, 1]; if false, returns smoothing kernel
/// * `dtype` - Output dtype
///
/// # Returns
///
/// 1D kernel tensor for use in separable convolution
///
/// # Example
///
/// ```ignore
/// // For Sobel filter along x-axis:
/// // - Use derivative kernel in x direction
/// // - Use smoothing kernel in y direction
/// let deriv = edge_kernel_1d(&client, "sobel", true, DType::F64)?;   // [-1, 0, 1]
/// let smooth = edge_kernel_1d(&client, "sobel", false, DType::F64)?;  // [1, 2, 1]
/// ```
pub fn edge_kernel_1d<R, C>(
    client: &C,
    kind: &str,
    derivative: bool,
    dtype: DType,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + UnaryOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    if derivative {
        // Derivative kernel: [-1, 0, 1]
        // Create as [0, 1, 2] then subtract 1
        let positions = client.arange(0.0, 3.0, 1.0, dtype)?;
        client.add_scalar(&positions, -1.0)
    } else {
        match kind {
            "sobel" => {
                // Sobel smoothing: [1, 2, 1]
                // Create as [0, 1, 2] then convert using: 2 - abs(x - 1)
                let positions = client.arange(0.0, 3.0, 1.0, dtype)?;
                let ones = client.fill(&[3], 1.0, dtype)?;
                let centered = client.sub(&positions, &ones)?; // [-1, 0, 1]
                let abs_val = client.abs(&centered)?; // [1, 0, 1]
                let scaled = client.mul_scalar(&abs_val, -1.0)?; // [-1, 0, -1]
                client.add_scalar(&scaled, 2.0) // [1, 2, 1]
            }
            "prewitt" => {
                // Prewitt smoothing: [1, 1, 1] - just all ones
                client.fill(&[3], 1.0, dtype)
            }
            _ => Err(Error::InvalidArgument {
                arg: "kind",
                reason: format!("Unknown edge kernel kind: {kind}. Use 'sobel' or 'prewitt'"),
            }),
        }
    }
}

/// Generate a 1D Laplacian kernel: [1, -2, 1].
///
/// Used for computing the discrete second derivative. For N-D Laplacian,
/// apply this kernel separably along each axis.
///
/// # Arguments
///
/// * `client` - Runtime client for tensor creation
/// * `dtype` - Output dtype (F32 or F64)
///
/// # Returns
///
/// 1D Laplacian kernel [1, -2, 1]
pub fn laplace_kernel_1d<R, C>(client: &C, dtype: DType) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + UnaryOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    // Create [1, -2, 1] kernel
    // Strategy: [1, 1, 1] - [0, 3, 0] = [1, -2, 1]
    // To get [0, 3, 0]: compute 3 * (1 - abs(x-1))
    //
    // positions [0, 1, 2]
    // centered = positions - 1 = [-1, 0, 1]
    // abs_centered = [1, 0, 1]
    // offset = 3 * (1 - abs_centered) = 3 * [0, 1, 0] = [0, 3, 0]
    // result = [1,1,1] - [0,3,0] = [1, -2, 1] ✓

    let positions = client.arange(0.0, 3.0, 1.0, dtype)?;
    let ones = client.fill(&[3], 1.0, dtype)?;
    let centered = client.sub(&positions, &ones)?; // [-1, 0, 1]
    let abs_centered = client.abs(&centered)?; // [1, 0, 1]

    // Compute 1 - abs_centered = [0, 1, 0]
    let one_minus_abs = client.sub(&ones, &abs_centered)?; // [0, 1, 0]
    // Multiply by 3 to get [0, 3, 0]
    let offset = client.mul_scalar(&one_minus_abs, 3.0)?; // [0, 3, 0]

    // Create base [1, 1, 1]
    let base = client.fill(&[3], 1.0, dtype)?;
    // Compute [1, 1, 1] - [0, 3, 0] = [1, -2, 1]
    client.sub(&base, &offset)
}

#[cfg(test)]
mod tests {

    // Note: Full numerical testing requires instantiating a runtime client.
    // These are compile-time structure tests only.

    #[test]
    fn test_gaussian_kernel_size_calculation() {
        // With sigma=1.0, truncate=4.0, radius should be ~4
        // size should be 2*4+1 = 9
        let radius = (4.0 * 1.0 + 0.5) as usize;
        assert_eq!(2 * radius + 1, 9);
    }

    #[test]
    fn test_laplace_formula_correctness() {
        // Verify the laplace kernel formula: [1, -2, 1] = [1,1,1] - [0,3,0]
        let base = [1.0, 1.0, 1.0];
        let offset = [0.0, 3.0, 0.0];
        let result: [f64; 3] = [
            base[0] - offset[0],
            base[1] - offset[1],
            base[2] - offset[2],
        ];
        assert_eq!(result, [1.0, -2.0, 1.0]);
    }

    #[test]
    fn test_sobel_formula_correctness() {
        // Verify sobel kernel formula: 2 - abs(x-1) gives [1, 2, 1]
        let xs = [0.0_f64, 1.0_f64, 2.0_f64];
        let result: Vec<f64> = xs.iter().map(|&x| 2.0 - (x - 1.0).abs()).collect();
        assert_eq!(result, vec![1.0, 2.0, 1.0]);
    }
}
