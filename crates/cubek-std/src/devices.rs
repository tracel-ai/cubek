//! Devices a dev machine cannot show you, as functions a routine's geometry is unit-tested
//! against, and the few changes a test makes to one.

use std::sync::Arc;

use cubecl::features::{
    AtomicUsage, Features, MmaConfig, Plane, ScaledMmaConfig, Tma, WgmmaConfig,
};
use cubecl::ir::{
    DeviceIdentity, DeviceProperties, ElemType, FloatKind, HardwareProperties,
    MemoryDeviceProperties, OpaqueType, Type,
};
use cubecl::profile::TimingMethod;

use crate::Hardware;

/// The dev machine's AMD Radeon 8060S (Strix Halo, gfx1151): wave32 planes, 1024 units per
/// cube, 64 KiB shared memory.
pub fn strix_halo() -> Hardware {
    let mut device = device(HardwareProperties {
        num_streaming_multiprocessors: Some(40),
        min_tensor_cores_dim: Some(16),
        max_vector_size: 8,
        ..gpu(32, 64 * 1024)
    });
    device.features.plane = Plane::Ops | Plane::NonUniformControlFlow;
    device.features.device_memory_scope = true;
    f32_atomic_add(&mut device);
    shared_f32_atomic_add(&mut device);
    hardware(device)
}

/// A WebGPU adapter with no subgroup support, reporting the plane width the wgpu runtime
/// fabricates for exactly that case: the device that turns `plane_size_max > 1` into silent
/// garbage.
pub fn wgpu_no_subgroups() -> Hardware {
    hardware(device(HardwareProperties {
        plane_size_min: 8,
        plane_size_max: 128,
        max_units_per_cube: 256,
        max_cube_dim: (256, 256, 64),
        max_cube_count: (65535, 65535, 65535),
        ..gpu(32, 16 * 1024)
    }))
}

/// An adapter that supports planes but reports a range: AMD RDNA, which runs wave32 or wave64
/// depending on the shader.
pub fn rdna_range() -> Hardware {
    let mut device = device(HardwareProperties {
        plane_size_max: 64,
        num_streaming_multiprocessors: Some(40),
        cube_mma_reserved_shared_memory: 16 * 1024,
        ..gpu(32, 64 * 1024)
    });
    device.features.plane = Plane::Ops.into();
    device.features.device_memory_scope = true;
    hardware(device)
}

/// AMD CDNA (MI300-class) under ROCm: an exact wave64, which halves the planes a cube can hold
/// and doubles what each split of a shared reduction costs.
pub fn cdna_wave64() -> Hardware {
    let mut device = device(HardwareProperties {
        num_streaming_multiprocessors: Some(228),
        min_tensor_cores_dim: Some(16),
        ..gpu(64, 64 * 1024)
    });
    device.features.plane = Plane::Ops | Plane::NonUniformControlFlow;
    device.features.device_memory_scope = true;
    f32_atomic_add(&mut device);
    shared_f32_atomic_add(&mut device);
    hardware(device)
}

/// Apple M2 Pro (Metal via MSL): wave32 planes with non-uniform control flow, 1024-unit
/// cubes, 32 KiB shared memory, Metal's 30-binding cap, no TMA.
pub fn m2_pro() -> Hardware {
    let mut device = device(HardwareProperties {
        num_streaming_multiprocessors: Some(19),
        max_bindings: 30,
        ..gpu(32, 32 * 1024)
    });
    device.features.plane = Plane::Ops | Plane::NonUniformControlFlow;
    device.features.device_memory_scope = true;
    f32_atomic_add(&mut device);
    hardware(device)
}

/// Hopper (H100): TMA, async copies and barriers, big shared memory. It offers no instruction
/// of its own: a test that needs one states it.
pub fn h100() -> Hardware {
    let mut device = device(HardwareProperties {
        num_streaming_multiprocessors: Some(132),
        num_tensor_cores: Some(4),
        min_tensor_cores_dim: Some(16),
        max_cube_dim: (1024, 1024, 64),
        max_cube_count: (2147483647, 65535, 65535),
        max_bindings: 32,
        ..gpu(32, 228 * 1024)
    });
    device.features.plane = Plane::Ops | Plane::NonUniformControlFlow;
    device.features.device_memory_scope = true;
    device.features.tma = Tma::Base.into();
    device.features.copy_async = true;
    device.features.types.opaque.insert(OpaqueType::Barrier);
    f32_atomic_add(&mut device);
    shared_f32_atomic_add(&mut device);
    Hardware {
        accumulator_as_a: true,
        ..hardware(device)
    }
}

/// A 12-core CPU runtime, as cubecl's `CpuRuntime` reports it: single-unit planes with no
/// plane ops, a cube of one unit per core on every axis, its L1d as shared memory, reads sized
/// to four 128-bit registers back to back, and unbounded vectors.
pub fn cpu_12() -> Hardware {
    let mut device = device(HardwareProperties {
        plane_size_min: 1,
        plane_size_max: 1,
        max_units_per_cube: 12,
        max_cube_dim: (12, 12, 12),
        num_cpu_cores: Some(12),
        last_level_cache_size: Some(32 * 1024 * 1024),
        max_vector_size: usize::MAX,
        ..gpu(1, 64 * 1024)
    });
    device.io_width_override = Some(512);
    f32_atomic_add(&mut device);
    shared_f32_atomic_add(&mut device);
    hardware(device)
}

/// `hw` offering exactly `shapes` through the vendor's fragment API (`cmma`).
pub fn offering_cmma(mut hw: Hardware, shapes: impl IntoIterator<Item = MmaConfig>) -> Hardware {
    features(&mut hw).matmul.cmma = shapes.into_iter().collect();
    hw
}

/// `hw` offering exactly `shapes` through the matrix intrinsics (`mma`).
pub fn offering_mma(mut hw: Hardware, shapes: impl IntoIterator<Item = MmaConfig>) -> Hardware {
    features(&mut hw).matmul.mma = shapes.into_iter().collect();
    hw
}

/// `hw` offering exactly `shapes` as block-scaled instructions.
pub fn offering_scaled_mma(
    mut hw: Hardware,
    shapes: impl IntoIterator<Item = ScaledMmaConfig>,
) -> Hardware {
    features(&mut hw).matmul.scaled_mma = shapes.into_iter().collect();
    hw
}

/// `hw` offering Blackwell's block-scaled instruction over NVFP4: `16 × 8 × 64`, `e2m1`
/// pairs under an `e4m3` scale every sixteen values of `k`, summed in `f32`.
pub fn offering_nvfp4(hw: Hardware) -> Hardware {
    let fp4 = ElemType::Float(FloatKind::E2M1x2);
    offering_scaled_mma(
        hw,
        [ScaledMmaConfig {
            a_type: fp4,
            b_type: fp4,
            cd_type: ElemType::Float(FloatKind::F32),
            scales_type: ElemType::Float(FloatKind::E4M3),
            m: 16,
            n: 8,
            k: 64,
            scales_factor: 4,
        }],
    )
}

/// `hw` offering exactly `forms` as warpgroup MMAs.
pub fn offering_wgmma(mut hw: Hardware, forms: impl IntoIterator<Item = WgmmaConfig>) -> Hardware {
    features(&mut hw).matmul.wgmma = forms.into_iter().collect();
    hw
}

/// `hw`'s topology, for a test to change one figure of.
pub fn topology(hw: &mut Hardware) -> &mut HardwareProperties {
    &mut Arc::make_mut(&mut hw.properties).hardware
}

/// `hw`'s features, for a test to change one capability of.
pub fn features(hw: &mut Hardware) -> &mut Features {
    &mut Arc::make_mut(&mut hw.properties).features
}

/// The figures every GPU fixture shares, at `plane_size` and `shared_memory` bytes.
fn gpu(plane_size: u32, shared_memory: usize) -> HardwareProperties {
    HardwareProperties {
        load_width: 128,
        vector_register_count: None,
        plane_size_min: plane_size,
        plane_size_max: plane_size,
        max_bindings: 8,
        max_shared_memory_size: shared_memory,
        max_cube_count: (u32::MAX, u32::MAX, u32::MAX),
        max_units_per_cube: 1024,
        max_cube_dim: (1024, 1024, 1024),
        num_streaming_multiprocessors: None,
        num_cpu_cores: None,
        last_level_cache_size: None,
        num_tensor_cores: None,
        min_tensor_cores_dim: None,
        max_vector_size: 4,
        cube_mma_reserved_shared_memory: 0,
    }
}

/// A device of `hardware` with no feature, which each fixture then grants.
fn device(hardware: HardwareProperties) -> DeviceProperties {
    DeviceProperties::new(
        Features::default(),
        MemoryDeviceProperties::new(u64::MAX, 4),
        hardware,
        TimingMethod::System,
        DeviceIdentity {
            name: "fixture".into(),
            fingerprint: "fixture".into(),
            physical: None,
        },
    )
}

fn hardware(device: DeviceProperties) -> Hardware {
    Hardware {
        properties: Arc::new(device),
        accumulator_as_a: false,
    }
}

fn f32_atomic_add(device: &mut DeviceProperties) {
    device.register_atomic_type_usage(
        Type::atomic(ElemType::Float(FloatKind::F32)),
        AtomicUsage::Add,
    );
}

fn shared_f32_atomic_add(device: &mut DeviceProperties) {
    device.register_shared_atomic_type_usage(
        Type::atomic(ElemType::Float(FloatKind::F32)),
        AtomicUsage::Add,
    );
}
