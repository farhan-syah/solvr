//! CUDA implementation of statistical algorithms.
//!
//! This module implements the [`StatisticsAlgorithms`] trait for CUDA
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
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl StatisticsAlgorithms<CudaRuntime> for CudaClient {
    fn describe(&self, x: &Tensor<CudaRuntime>) -> Result<TensorDescriptiveStats<CudaRuntime>> {
        describe_impl(self, x)
    }

    fn percentile(&self, x: &Tensor<CudaRuntime>, p: f64) -> Result<Tensor<CudaRuntime>> {
        percentile_impl(self, x, p)
    }

    fn iqr(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        iqr_impl(self, x)
    }

    fn skewness(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        skewness_impl(self, x)
    }

    fn kurtosis(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        kurtosis_impl(self, x)
    }

    fn zscore(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        zscore_impl(self, x)
    }

    fn sem(&self, x: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        sem_impl(self, x)
    }

    fn ttest_1samp(
        &self,
        x: &Tensor<CudaRuntime>,
        popmean: f64,
    ) -> Result<TensorTestResult<CudaRuntime>> {
        ttest_1samp_impl(self, x, popmean)
    }

    fn ttest_ind(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<TensorTestResult<CudaRuntime>> {
        ttest_ind_impl(self, a, b)
    }

    fn ttest_rel(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<TensorTestResult<CudaRuntime>> {
        ttest_rel_impl(self, a, b)
    }

    fn pearsonr(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
    ) -> Result<TensorTestResult<CudaRuntime>> {
        pearsonr_impl(self, x, y)
    }

    fn spearmanr(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
    ) -> Result<TensorTestResult<CudaRuntime>> {
        spearmanr_impl(self, x, y)
    }

    fn linregress(
        &self,
        x: &Tensor<CudaRuntime>,
        y: &Tensor<CudaRuntime>,
    ) -> Result<LinregressResult> {
        linregress_impl(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stats::helpers::extract_scalar;
    use numr::runtime::cuda::CudaDevice;

    fn setup() -> Option<(CudaClient, CudaDevice)> {
        // Skip if no CUDA device available
        let device = CudaDevice::new(0).ok()?;
        let client = CudaClient::new(device.clone());
        Some((client, device))
    }

    #[test]
    fn test_describe_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let data = Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let stats = client.describe(&data).unwrap();

        assert_eq!(stats.nobs, 5);

        let mean_val = extract_scalar(&stats.mean).unwrap();
        assert!((mean_val - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ttest_1samp_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let data = Tensor::<CudaRuntime>::from_slice(&[1.2f64, 1.5, 1.3, 1.4, 1.6], &[5], &device);
        let result = client.ttest_1samp(&data, 1.0).unwrap();

        let stat = extract_scalar(&result.statistic).unwrap();
        let pval = extract_scalar(&result.pvalue).unwrap();

        assert!(stat > 0.0);
        assert!(pval < 0.05);
    }

    #[test]
    fn test_linregress_cuda() {
        let Some((client, device)) = setup() else {
            eprintln!("Skipping CUDA test: no device available");
            return;
        };

        let x = Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        let y = Tensor::<CudaRuntime>::from_slice(&[2.0f64, 4.0, 6.0, 8.0, 10.0], &[5], &device);

        let result = client.linregress(&x, &y).unwrap();

        assert!((result.slope - 2.0).abs() < 1e-10);
        assert!((result.intercept - 0.0).abs() < 1e-10);
    }
}
