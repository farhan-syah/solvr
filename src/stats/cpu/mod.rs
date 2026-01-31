//! CPU implementation of statistical algorithms.
//!
//! This module implements the [`StatisticsAlgorithms`] trait for CPU
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
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl StatisticsAlgorithms<CpuRuntime> for CpuClient {
    fn describe(&self, x: &Tensor<CpuRuntime>) -> Result<TensorDescriptiveStats<CpuRuntime>> {
        describe_impl(self, x)
    }

    fn percentile(&self, x: &Tensor<CpuRuntime>, p: f64) -> Result<Tensor<CpuRuntime>> {
        percentile_impl(self, x, p)
    }

    fn iqr(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        iqr_impl(self, x)
    }

    fn skewness(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        skewness_impl(self, x)
    }

    fn kurtosis(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        kurtosis_impl(self, x)
    }

    fn zscore(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        zscore_impl(self, x)
    }

    fn sem(&self, x: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        sem_impl(self, x)
    }

    fn ttest_1samp(
        &self,
        x: &Tensor<CpuRuntime>,
        popmean: f64,
    ) -> Result<TensorTestResult<CpuRuntime>> {
        ttest_1samp_impl(self, x, popmean)
    }

    fn ttest_ind(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<TensorTestResult<CpuRuntime>> {
        ttest_ind_impl(self, a, b)
    }

    fn ttest_rel(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<TensorTestResult<CpuRuntime>> {
        ttest_rel_impl(self, a, b)
    }

    fn pearsonr(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
    ) -> Result<TensorTestResult<CpuRuntime>> {
        pearsonr_impl(self, x, y)
    }

    fn spearmanr(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
    ) -> Result<TensorTestResult<CpuRuntime>> {
        spearmanr_impl(self, x, y)
    }

    fn linregress(
        &self,
        x: &Tensor<CpuRuntime>,
        y: &Tensor<CpuRuntime>,
    ) -> Result<LinregressResult> {
        linregress_impl(self, x, y)
    }
}

#[cfg(test)]
mod tests;
