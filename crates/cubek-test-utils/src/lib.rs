//! What cubek's tests and benchmarks share: inputs built on the device with their values kept on
//! the host ([`TestInput`], [`TileInput`], [`QuantizedTileInput`]), the comparison of a kernel's
//! output with a reference under the test policy `cubek.toml` sets ([`assert_equals_approx`],
//! [`TestOutcome`]), one shape for testing a kernel or a microkernel against its own reference
//! ([`ScopeChecker`]), and the catalogue a benchmark harness runs ([`Category`]), whose rows
//! report their progress through a [`Progress`].

mod config;
mod correctness;
pub mod hardware;
mod progress;
pub mod quant_layout;
mod registry;
mod scope;
mod stubs;
mod test_mode;
mod test_tensor;
mod tile_input;

pub use config::{CubekConfig, PrintSection, PrintView, TestPolicy, TestSection, config};
pub use correctness::{
    DimFilter, HostAttention, PackedVectors, TensorFilter, assert_equals_approx,
    assert_equals_approx_in_slice, compare_host_data_files, parse_tensor_filter, print_tensor,
    print_tensors,
};
pub use progress::Progress;
pub use registry::{
    BenchmarkCategory, Binding, CatalogEntry, Category, CategoryWork, ComputeWork, Correctness,
    ItemDescriptor, ResourceKind, RunSamples, client, profile_launch, supports_f16_arithmetic,
    timing_method,
};
pub use scope::*;
pub use test_mode::*;
pub use test_tensor::*;
pub use tile_input::*;
