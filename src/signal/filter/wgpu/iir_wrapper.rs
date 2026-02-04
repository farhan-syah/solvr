//! WebGPU implementation of unified IIR filter design.

// Allow many arguments for filter design functions that match scipy's signature
#![allow(clippy::too_many_arguments)]

use crate::signal::filter::impl_generic::iirfilter_impl;
use crate::signal::filter::traits::iir_design::IirDesignResult;
use crate::signal::filter::traits::iir_wrapper::{IirDesignType, IirFilterAlgorithms};
use crate::signal::filter::types::{FilterOutput, FilterType};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};

impl IirFilterAlgorithms<WgpuRuntime> for WgpuClient {
    fn iirfilter(
        &self,
        order: usize,
        wn: &[f64],
        filter_type: FilterType,
        design_type: IirDesignType,
        rp: Option<f64>,
        rs: Option<f64>,
        output: FilterOutput,
        device: &<WgpuRuntime as numr::runtime::Runtime>::Device,
    ) -> Result<IirDesignResult<WgpuRuntime>> {
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
