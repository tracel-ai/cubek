pub mod components;
pub mod definition;
pub mod kernels;
pub mod launch;
pub mod routines;

#[cfg(any(feature = "cpu-reference", feature = "benchmarks"))]
pub mod eval;

// Re-export per-operation modules at the crate root for internal paths
// (`crate::forward`, etc.) and for downstream users that previously relied on
// `cubek_convolution::*`.
pub use kernels::{backward_data, backward_weight, forward};

// Top-level launcher: the single public entry point.
pub use launch::{
    AcceleratedTileKind, ConvAlgorithm, ConvolutionArgs, ConvolutionInputs, Strategy, launch_ref,
};

// The depthwise routine is its own entry point rather than a `ConvAlgorithm`: it shares no
// blueprint with the accelerated routines, so there is nothing for `Strategy` to select.
pub use kernels::forward::{
    DepthwiseStrategy, DepthwiseTensors, DepthwiseTiling, launch_depthwise,
};

// The direct routine is its own entry point for the same reason depthwise is: no stage
// hierarchy, so there is nothing for a blueprint to size. It is what every CPU convolution runs.
pub use kernels::forward::{DirectTensors, launch_direct};
