//! Filter algorithm traits.

pub mod analog_response;
pub mod conversions;
pub mod fir_design;
pub mod iir_design;
pub mod iir_wrapper;
pub mod lti_system;
pub mod state_space;
pub mod system_response;

pub use analog_response::{AnalogResponseAlgorithms, FreqsResult};
pub use conversions::FilterConversions;
pub use fir_design::FirDesignAlgorithms;
pub use iir_design::IirDesignAlgorithms;
pub use iir_wrapper::{IirDesignType, IirFilterAlgorithms};
pub use lti_system::{DiscreteTimeLtiAlgorithms, DlsimResult};
pub use state_space::StateSpaceConversions;
pub use system_response::{ImpulseResponse, StepResponse, SystemResponseAlgorithms};
