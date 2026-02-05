//! Trait definitions and types for integration algorithms.

mod algorithms;
pub mod events;
mod types;

pub use algorithms::IntegrationAlgorithms;
pub use events::EventFunction;
pub use types::{
    BVPResult, MonteCarloMethod, MonteCarloOptions, MonteCarloResult, NQuadOptions, QMCOptions,
    QMCSequence, QuadOptions, QuadResult, RombergOptions, SymplecticResult, TanhSinhOptions,
};
