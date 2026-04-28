pub mod components;
pub mod definition;
pub mod kernels;
pub mod launch;
pub mod routines;

// Re-export per-operation modules at the crate root for internal paths
// (`crate::forward`, etc.) and for downstream users that previously relied on
// `cubek_convolution::*`.
pub use kernels::{backward_data, backward_weight, forward};

// Top-level launcher: the single public entry point.
pub use launch::{
    AcceleratedTileKind, ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy, launch_ref,
};
