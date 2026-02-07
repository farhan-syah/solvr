//! Statistical algorithm traits and types.

mod descriptive;
mod hypothesis;
mod information;
mod regression;
mod robust;
mod types;

pub use descriptive::DescriptiveStatisticsAlgorithms;
pub use hypothesis::HypothesisTestingAlgorithms;
pub use information::InformationTheoryAlgorithms;
pub use regression::RegressionAlgorithms;
pub use robust::RobustStatisticsAlgorithms;
pub use types::{
    LeveneCenter, LinregressResult, RobustRegressionResult, TensorDescriptiveStats,
    TensorTestResult, validate_stats_dtype,
};
