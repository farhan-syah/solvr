//! CPU implementation of unified IIR filter design.

// Allow many arguments for filter design functions that match scipy's signature
#![allow(clippy::too_many_arguments)]

use crate::signal::filter::impl_generic::iirfilter_impl;
use crate::signal::filter::traits::iir_design::IirDesignResult;
use crate::signal::filter::traits::iir_wrapper::{IirDesignType, IirFilterAlgorithms};
use crate::signal::filter::types::{FilterOutput, FilterType};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};

impl IirFilterAlgorithms<CpuRuntime> for CpuClient {
    fn iirfilter(
        &self,
        order: usize,
        wn: &[f64],
        filter_type: FilterType,
        design_type: IirDesignType,
        rp: Option<f64>,
        rs: Option<f64>,
        output: FilterOutput,
        device: &<CpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<IirDesignResult<CpuRuntime>> {
        iirfilter_impl(
            self,
            order,
            wn,
            filter_type,
            design_type,
            rp,
            rs,
            output,
            device,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::filter::IirDesignAlgorithms;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_iirfilter_butter() {
        let (client, device) = setup();

        // Compare iirfilter with direct butter call
        let result1 = client
            .iirfilter(
                4,
                &[0.2],
                FilterType::Lowpass,
                IirDesignType::Butterworth,
                None,
                None,
                FilterOutput::Ba,
                &device,
            )
            .unwrap();

        let result2 = client
            .butter(4, &[0.2], FilterType::Lowpass, FilterOutput::Ba, &device)
            .unwrap();

        let tf1 = result1.as_ba().unwrap();
        let tf2 = result2.as_ba().unwrap();

        let b1: Vec<f64> = tf1.b.to_vec();
        let b2: Vec<f64> = tf2.b.to_vec();

        for (v1, v2) in b1.iter().zip(b2.iter()) {
            assert!((v1 - v2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_iirfilter_cheby1() {
        let (client, device) = setup();

        // Compare iirfilter with direct cheby1 call
        let result1 = client
            .iirfilter(
                3,
                &[0.25],
                FilterType::Lowpass,
                IirDesignType::Chebyshev1,
                Some(1.0),
                None,
                FilterOutput::Ba,
                &device,
            )
            .unwrap();

        let result2 = client
            .cheby1(
                3,
                1.0,
                &[0.25],
                FilterType::Lowpass,
                FilterOutput::Ba,
                &device,
            )
            .unwrap();

        let tf1 = result1.as_ba().unwrap();
        let tf2 = result2.as_ba().unwrap();

        let b1: Vec<f64> = tf1.b.to_vec();
        let b2: Vec<f64> = tf2.b.to_vec();

        for (v1, v2) in b1.iter().zip(b2.iter()) {
            assert!((v1 - v2).abs() < 1e-10);
        }
    }

    #[test]
    fn test_iirfilter_cheby2() {
        let (client, device) = setup();

        let result = client
            .iirfilter(
                3,
                &[0.25],
                FilterType::Lowpass,
                IirDesignType::Chebyshev2,
                None,
                Some(40.0),
                FilterOutput::Sos,
                &device,
            )
            .unwrap();

        let sos = result.as_sos().unwrap();
        assert!(sos.num_sections() >= 1);
    }

    #[test]
    fn test_iirfilter_elliptic() {
        let (client, device) = setup();

        let result = client
            .iirfilter(
                3,
                &[0.25],
                FilterType::Lowpass,
                IirDesignType::Elliptic,
                Some(1.0),
                Some(40.0),
                FilterOutput::Zpk,
                &device,
            )
            .unwrap();

        let zpk = result.as_zpk().unwrap();
        assert!(zpk.num_poles() > 0);
    }

    #[test]
    fn test_iirfilter_bessel() {
        let (client, device) = setup();

        let result = client
            .iirfilter(
                4,
                &[0.2],
                FilterType::Lowpass,
                IirDesignType::Bessel,
                None,
                None,
                FilterOutput::Ba,
                &device,
            )
            .unwrap();

        let tf = result.as_ba().unwrap();
        assert_eq!(tf.b.shape()[0], 5);
        assert_eq!(tf.a.shape()[0], 5);
    }

    #[test]
    fn test_iirfilter_bandpass() {
        let (client, device) = setup();

        let result = client
            .iirfilter(
                2,
                &[0.2, 0.4],
                FilterType::Bandpass,
                IirDesignType::Butterworth,
                None,
                None,
                FilterOutput::Zpk,
                &device,
            )
            .unwrap();

        let zpk = result.as_zpk().unwrap();
        // Bandpass doubles the order
        assert_eq!(zpk.num_poles(), 4);
    }

    #[test]
    fn test_iirfilter_missing_rp() {
        let (client, device) = setup();

        // Chebyshev1 requires rp
        let result = client.iirfilter(
            3,
            &[0.25],
            FilterType::Lowpass,
            IirDesignType::Chebyshev1,
            None,
            None,
            FilterOutput::Ba,
            &device,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_iirfilter_missing_rs() {
        let (client, device) = setup();

        // Chebyshev2 requires rs
        let result = client.iirfilter(
            3,
            &[0.25],
            FilterType::Lowpass,
            IirDesignType::Chebyshev2,
            None,
            None,
            FilterOutput::Ba,
            &device,
        );

        assert!(result.is_err());
    }
}
