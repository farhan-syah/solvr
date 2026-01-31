//! WebGPU implementation of statistical algorithms.
//!
//! This module implements the [`StatisticsAlgorithms`] trait for WebGPU
//! using numr's tensor operations via the generic implementation.

use crate::stats::impl_generic::{
    describe_impl, iqr_impl, kurtosis_impl, linregress_impl, pearsonr_impl, percentile_impl,
    sem_impl, skewness_impl, spearmanr_impl, ttest_1samp_impl, ttest_ind_impl, ttest_rel_impl,
    zscore_impl,
};
use crate::stats::{
    LinregressResult, StatisticsAlgorithms, TensorDescriptiveStats, TensorTestResult,
};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl StatisticsAlgorithms<WgpuRuntime> for WgpuClient {
    fn describe(&self, x: &Tensor<WgpuRuntime>) -> Result<TensorDescriptiveStats<WgpuRuntime>> {
        describe_impl(self, x)
    }

    fn percentile(&self, x: &Tensor<WgpuRuntime>, p: f64) -> Result<Tensor<WgpuRuntime>> {
        percentile_impl(self, x, p)
    }

    fn iqr(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        iqr_impl(self, x)
    }

    fn skewness(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        skewness_impl(self, x)
    }

    fn kurtosis(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        kurtosis_impl(self, x)
    }

    fn zscore(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        zscore_impl(self, x)
    }

    fn sem(&self, x: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        sem_impl(self, x)
    }

    fn ttest_1samp(
        &self,
        x: &Tensor<WgpuRuntime>,
        popmean: f64,
    ) -> Result<TensorTestResult<WgpuRuntime>> {
        ttest_1samp_impl(self, x, popmean)
    }

    fn ttest_ind(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<TensorTestResult<WgpuRuntime>> {
        ttest_ind_impl(self, a, b)
    }

    fn ttest_rel(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<TensorTestResult<WgpuRuntime>> {
        ttest_rel_impl(self, a, b)
    }

    fn pearsonr(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
    ) -> Result<TensorTestResult<WgpuRuntime>> {
        pearsonr_impl(self, x, y)
    }

    fn spearmanr(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
    ) -> Result<TensorTestResult<WgpuRuntime>> {
        spearmanr_impl(self, x, y)
    }

    fn linregress(
        &self,
        x: &Tensor<WgpuRuntime>,
        y: &Tensor<WgpuRuntime>,
    ) -> Result<LinregressResult> {
        linregress_impl(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::wgpu::WgpuDevice;

    fn setup() -> Option<(WgpuClient, WgpuDevice)> {
        // Skip if no WebGPU device available
        let device = WgpuDevice::new().ok()?;
        let client = WgpuClient::new(device.clone());
        Some((client, device))
    }

    #[test]
    fn test_describe_wgpu() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping WebGPU test: no device available");
            return;
        };

        let data = Tensor::<WgpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let stats = client.describe(&data).unwrap();

        assert_eq!(stats.nobs, 5);

        let mean_val = extract_scalar(&stats.mean).unwrap();
        assert!((mean_val - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ttest_1samp_wgpu() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping WebGPU test: no device available");
            return;
        };

        let data = Tensor::<WgpuRuntime>::from_slice(&[1.2f64, 1.5, 1.3, 1.4, 1.6], &[5], &device);
        let result = client.ttest_1samp(&data, 1.0).unwrap();

        let stat = extract_scalar(&result.statistic).unwrap();
        let pval = extract_scalar(&result.pvalue).unwrap();

        assert!(stat > 0.0);
        assert!(pval < 0.05);
    }

    #[test]
    fn test_linregress_wgpu() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping WebGPU test: no device available");
            return;
        };

        let x = Tensor::<WgpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let y = Tensor::<WgpuRuntime>::from_slice(&[2.0f64, 4.0, 6.0, 8.0, 10.0], &[5], &device);

        let result = client.linregress(&x, &y).unwrap();

        assert!((result.slope - 2.0).abs() < 1e-10);
        assert!((result.intercept - 0.0).abs() < 1e-10);
    }
}
