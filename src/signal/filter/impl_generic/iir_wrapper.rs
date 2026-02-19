//! Generic implementation of unified IIR filter design.
//!
//! Dispatches to the appropriate design function based on filter type.

// Allow many arguments for filter design functions that match scipy's signature
#![allow(clippy::too_many_arguments)]
use crate::DType;

use super::prototypes::{butter_impl, cheby1_impl, cheby2_impl, design_iir_filter, ellip_impl};
use crate::signal::filter::traits::conversions::FilterConversions;
use crate::signal::filter::traits::iir_design::{BesselNorm, IirDesignResult};
use crate::signal::filter::traits::iir_wrapper::IirDesignType;
use crate::signal::filter::types::{FilterOutput, FilterType};
use numr::algorithm::polynomial::PolynomialAlgorithms;
use numr::error::{Error, Result};
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};

use super::prototypes::besselap_impl;

/// Unified IIR filter design implementation.
///
/// Dispatches to the appropriate design function based on `design_type`.
pub fn iirfilter_impl<R, C>(
    client: &C,
    order: usize,
    wn: &[f64],
    filter_type: FilterType,
    design_type: IirDesignType,
    rp: Option<f64>,
    rs: Option<f64>,
    output: FilterOutput,
    device: &R::Device,
) -> Result<IirDesignResult<R>>
where
    R: Runtime<DType = DType>,
    C: FilterConversions<R>
        + PolynomialAlgorithms<R>
        + ScalarOps<R>
        + TensorOps<R>
        + RuntimeClient<R>,
{
    match design_type {
        IirDesignType::Butterworth => butter_impl(client, order, wn, filter_type, output, device),

        IirDesignType::Chebyshev1 => {
            let rp = rp.ok_or_else(|| Error::InvalidArgument {
                arg: "rp",
                reason: "Chebyshev Type I requires passband ripple (rp)".to_string(),
            })?;
            cheby1_impl(client, order, rp, wn, filter_type, output, device)
        }

        IirDesignType::Chebyshev2 => {
            let rs = rs.ok_or_else(|| Error::InvalidArgument {
                arg: "rs",
                reason: "Chebyshev Type II requires stopband attenuation (rs)".to_string(),
            })?;
            cheby2_impl(client, order, rs, wn, filter_type, output, device)
        }

        IirDesignType::Elliptic => {
            let rp = rp.ok_or_else(|| Error::InvalidArgument {
                arg: "rp",
                reason: "Elliptic filter requires passband ripple (rp)".to_string(),
            })?;
            let rs = rs.ok_or_else(|| Error::InvalidArgument {
                arg: "rs",
                reason: "Elliptic filter requires stopband attenuation (rs)".to_string(),
            })?;
            ellip_impl(client, order, rp, rs, wn, filter_type, output, device)
        }

        IirDesignType::Bessel => {
            // Bessel uses default Phase normalization
            let proto = besselap_impl(client, order, BesselNorm::Phase, device)?;
            design_iir_filter(client, proto, wn, filter_type, output, device)
        }
    }
}
