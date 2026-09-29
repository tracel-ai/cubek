//! Devices a routine's decisions are tested against, as their runtimes report them.
//!
//! A test that turns on one property names it where it overrides it:
//! `HardwareProperties { num_cpu_cores: Some(4), ..AVX2 }`.

use cubecl::ir::{HardwareProperties, VectorSize};

/// A 16-thread x86 host with AVX2: sixteen 256-bit vector registers and a 32 KiB L1d.
pub const AVX2: HardwareProperties = HardwareProperties {
    load_width: 256,
    vector_register_count: Some(16),
    plane_size_min: 1,
    plane_size_max: 1,
    max_bindings: u32::MAX,
    max_shared_memory_size: 32 * 1024,
    max_cube_count: (u32::MAX, u32::MAX, u32::MAX),
    max_units_per_cube: 16,
    max_cube_dim: (16, 16, 16),
    num_streaming_multiprocessors: None,
    num_cpu_cores: Some(16),
    last_level_cache_size: None,
    num_tensor_cores: None,
    min_tensor_cores_dim: None,
    max_vector_size: VectorSize::MAX,
    cube_mma_reserved_shared_memory: 0,
};

/// The same host with AVX-512: thirty-two 512-bit vector registers.
pub const AVX512: HardwareProperties = HardwareProperties {
    load_width: 512,
    vector_register_count: Some(32),
    ..AVX2
};

/// An 8-core ARM host with NEON: thirty-two 128-bit vector registers and a 128 KiB L1d.
pub const NEON: HardwareProperties = HardwareProperties {
    load_width: 128,
    vector_register_count: Some(32),
    max_shared_memory_size: 128 * 1024,
    max_units_per_cube: 8,
    max_cube_dim: (8, 8, 8),
    num_cpu_cores: Some(8),
    ..AVX2
};

/// A GPU with 32-unit planes, 1024 units per cube, 48 KiB of shared memory and 64 streaming
/// multiprocessors. Like every GPU runtime, it states no vector register count.
pub const GPU: HardwareProperties = HardwareProperties {
    load_width: 128,
    vector_register_count: None,
    plane_size_min: 32,
    plane_size_max: 32,
    max_bindings: 32,
    max_shared_memory_size: 48 * 1024,
    max_cube_count: (u32::MAX, u32::MAX, u32::MAX),
    max_units_per_cube: 1024,
    max_cube_dim: (1024, 1024, 64),
    num_streaming_multiprocessors: Some(64),
    num_cpu_cores: None,
    last_level_cache_size: None,
    num_tensor_cores: Some(4),
    min_tensor_cores_dim: Some(16),
    max_vector_size: VectorSize::MAX,
    cube_mma_reserved_shared_memory: 0,
};
