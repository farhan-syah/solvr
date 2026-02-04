//! Bilinear transform implementation.
//!
//! Converts analog (continuous-time) filters to digital (discrete-time) filters
//! using the bilinear transformation.

use crate::signal::filter::types::{AnalogPrototype, ZpkFilter};
use numr::error::Result;
use numr::ops::ScalarOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::f64::consts::PI;

/// Apply bilinear transform to convert analog filter to digital.
///
/// # Algorithm
///
/// The bilinear transform maps the s-plane to the z-plane via:
/// ```text
/// s = (2*fs) * (z - 1) / (z + 1)
/// ```
///
/// For poles and zeros, the transformation is:
/// ```text
/// z = (1 + s/(2*fs)) / (1 - s/(2*fs))
/// ```
///
/// # Frequency Warping
///
/// The bilinear transform warps frequencies according to:
/// ```text
/// ω_digital = 2 * arctan(ω_analog / (2*fs))
/// ```
///
/// Pre-warping is typically applied during filter design to compensate.
pub fn bilinear_zpk_impl<R, C>(
    _client: &C,
    analog: &AnalogPrototype<R>,
    fs: f64,
) -> Result<ZpkFilter<R>>
where
    R: Runtime,
    C: ScalarOps<R> + RuntimeClient<R>,
{
    let device = analog.zeros_real.device();
    let _dtype = analog.zeros_real.dtype();

    let n_zeros = analog.zeros_real.shape()[0];
    let n_poles = analog.poles_real.shape()[0];

    // Get analog poles and zeros
    let z_re: Vec<f64> = analog.zeros_real.to_vec();
    let z_im: Vec<f64> = analog.zeros_imag.to_vec();
    let p_re: Vec<f64> = analog.poles_real.to_vec();
    let p_im: Vec<f64> = analog.poles_imag.to_vec();

    let fs2 = 2.0 * fs;

    // Transform zeros: z_d = (1 + s/(2*fs)) / (1 - s/(2*fs))
    let mut digital_zeros_re = Vec::with_capacity(n_poles); // Will add zeros at -1
    let mut digital_zeros_im = Vec::with_capacity(n_poles);

    for i in 0..n_zeros {
        let (zd_re, zd_im) = bilinear_point(z_re[i], z_im[i], fs2);
        digital_zeros_re.push(zd_re);
        digital_zeros_im.push(zd_im);
    }

    // Add zeros at z = -1 to match the number of poles
    // (The bilinear transform maps s = infinity to z = -1)
    for _ in n_zeros..n_poles {
        digital_zeros_re.push(-1.0);
        digital_zeros_im.push(0.0);
    }

    // Transform poles
    let mut digital_poles_re = Vec::with_capacity(n_poles);
    let mut digital_poles_im = Vec::with_capacity(n_poles);

    for i in 0..n_poles {
        let (pd_re, pd_im) = bilinear_point(p_re[i], p_im[i], fs2);
        digital_poles_re.push(pd_re);
        digital_poles_im.push(pd_im);
    }

    // Compute gain transformation using the bilinear formula:
    // k_d = k_a * prod(2*fs - z_i) / prod(2*fs - p_i)
    // where the products are over the analog zeros and poles
    let mut gain = analog.gain;

    // Product over analog zeros: (2*fs - z_i)
    for i in 0..n_zeros {
        let re = fs2 - z_re[i];
        let im = -z_im[i];
        let mag = (re * re + im * im).sqrt();
        gain *= mag;
    }

    // Product over analog poles: 1 / (2*fs - p_i)
    for i in 0..n_poles {
        let re = fs2 - p_re[i];
        let im = -p_im[i];
        let mag = (re * re + im * im).sqrt();
        gain /= mag;
    }

    // For each zero added at -1, we divide by 2
    // (This comes from the fact that added zeros are at s = infinity,
    // and the bilinear transform maps infinity to -1 with a factor of 2*fs)
    for _ in n_zeros..n_poles {
        gain /= fs2;
    }

    // Make gain real and positive (take absolute value)
    gain = gain.abs();

    let digital_zeros_re_t =
        Tensor::from_slice(&digital_zeros_re, &[digital_zeros_re.len()], device);
    let digital_zeros_im_t =
        Tensor::from_slice(&digital_zeros_im, &[digital_zeros_im.len()], device);
    let digital_poles_re_t =
        Tensor::from_slice(&digital_poles_re, &[digital_poles_re.len()], device);
    let digital_poles_im_t =
        Tensor::from_slice(&digital_poles_im, &[digital_poles_im.len()], device);

    Ok(ZpkFilter::new(
        digital_zeros_re_t,
        digital_zeros_im_t,
        digital_poles_re_t,
        digital_poles_im_t,
        gain,
    ))
}

/// Apply bilinear transform to a single complex point.
///
/// s → z = (1 + s/fs2) / (1 - s/fs2)
fn bilinear_point(s_re: f64, s_im: f64, fs2: f64) -> (f64, f64) {
    // z = (1 + s/fs2) / (1 - s/fs2)
    // Let s/fs2 = a + bi
    let a = s_re / fs2;
    let b = s_im / fs2;

    // Numerator: 1 + a + bi
    let num_re = 1.0 + a;
    let num_im = b;

    // Denominator: 1 - a - bi
    let denom_re = 1.0 - a;
    let denom_im = -b;

    // Complex division
    let denom_mag_sq = denom_re * denom_re + denom_im * denom_im;

    if denom_mag_sq < 1e-30 {
        // Pole at infinity maps to z = -1
        return (-1.0, 0.0);
    }

    let z_re = (num_re * denom_re + num_im * denom_im) / denom_mag_sq;
    let z_im = (num_im * denom_re - num_re * denom_im) / denom_mag_sq;

    (z_re, z_im)
}

/// Pre-warp a digital frequency to analog frequency.
///
/// Used to compensate for frequency warping in the bilinear transform.
///
/// # Arguments
///
/// * `w_digital` - Normalized digital frequency (0 to 1, where 1 = Nyquist)
/// * `fs` - Sample rate
///
/// # Returns
///
/// Analog frequency in rad/s.
pub fn prewarp(w_digital: f64, fs: f64) -> f64 {
    // ω_analog = 2 * fs * tan(π * w_digital / 2)
    // But w_digital is normalized to Nyquist, so:
    // ω_analog = 2 * fs * tan(π * w_digital)
    // where w_digital is fraction of Nyquist (0 to 1)
    2.0 * fs * (PI * w_digital / 2.0).tan()
}
