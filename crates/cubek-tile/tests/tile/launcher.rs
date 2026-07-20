//! Unit tests for [`Launcher`]: geometry read off the concrete space, kernel space dynamic.

use cubecl::{
    TestRuntime,
    prelude::*,
    quant::scheme::{QuantLevel, QuantParam, QuantScheme, QuantStore, QuantValue},
};
use cubek_tile::{
    Axis, CubeAxis, Cut, Leaf, Schedule, Storage, StridedTileArgLaunch, Tiling, WalkOrder,
};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

#[test]
fn launcher_geometry_matches_concrete_space() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(1, 1, 64, 64, 16).launcher(&client);

    // X: 64/16 cube tiles, Y: 64/32, nothing on Z.
    match launch.cube_count() {
        CubeCount::Static(x, y, z) => assert_eq!((x, y, z), (4, 2, 1)),
        _ => panic!("launcher geometry should be static"),
    }
    // Planes: within a 16×32 cube tile, 2×4 leaves of 8×8.
    let plane_size = client.properties().hardware.plane_size_max;
    assert_eq!(launch.cube_dim(), CubeDim::new_2d(plane_size, 8));
}

#[test]
fn launcher_kernel_space_is_dynamic_concrete_is_not() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(1, 1, 64, 64, 16).launcher(&client);

    for axis in [M, N, K] {
        assert!(launch.space().is_dynamic(axis));
        assert!(!launch.concrete().is_dynamic(axis));
    }
    assert_eq!(launch.concrete().extent(M), 64);
}

/// The footgun the launcher removes: geometry read after `all_dynamic` has no extents to
/// read. Pinned so the two-step order stays a real constraint, not a stale comment.
#[test]
#[should_panic(expected = "Dynamic")]
fn geometry_after_dynamic_panics() {
    let _ = batched_space(1, 1, 64, 64, 16).all_dynamic().cube_count();
}

// ---- Launcher::arg ---------------------------------------------------------

const B0: Axis = Axis(3);
const B1: Axis = Axis(4);

fn binding(client: &ComputeClient<TestRuntime>, shape: &[usize]) -> TensorBinding<TestRuntime> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let len: usize = shape.iter().product();
    TensorBinding {
        handle: client.empty(len * size_of::<f32>()).binding(),
        strides: strides.into(),
        shape: shape.to_vec().into(),
        runtime: core::marker::PhantomData,
    }
}

/// A cpu_gemm-shaped scheme: two batch axes riding one-per-cube on Z, 16×32 cube tiles on
/// X/Y, 8×8 plane leaves with `leaf_k = 4`.
fn batched_space(b0: usize, b1: usize, m: usize, n: usize, k: usize) -> cubek_tile::Space {
    let batches = [B0, B1];
    Tiling::new()
        .extents(&[(B0, b0), (B1, b1), (M, m), (N, n), (K, k)])
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axes(&batches, Cut::cube(CubeAxis::Z, 1))
                .axis(M, Cut::cube(CubeAxis::X, 16))
                .axis(N, Cut::cube(CubeAxis::Y, 32))
                .axis(K, Cut::sequential(k))
        })
        .level(WalkOrder::RowMajor, Schedule::Direct, |l| {
            l.axes(&batches, Cut::sequential(1))
                .axis(M, Cut::plane(8))
                .axis(N, Cut::plane(8))
                .axis(K, Cut::sequential(4))
        })
        .leaf(Leaf::Register)
}

#[test]
fn arg_derives_check_from_subspace_overhang() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    // k = 18 overhangs its leaf (4); M and N divide everywhere.
    let launch = batched_space(1, 1, 64, 64, 18).launcher(&client);

    let touches_k = launch
        .arg::<f32>(binding(&client, &[64, 18]))
        .subspace(&[M, K])
        .build();
    assert!(touches_k.storage.check_bounds);

    let avoids_k = launch
        .arg::<f32>(binding(&client, &[64, 64]))
        .subspace(&[M, N])
        .build();
    assert!(!avoids_k.storage.check_bounds);

    // An explicit override still wins over the derivation.
    let forced = launch
        .arg::<f32>(binding(&client, &[64, 18]))
        .subspace(&[M, K])
        .checked(false)
        .build();
    assert!(!forced.storage.check_bounds);
}

#[test]
fn arg_right_aligns_batches_and_drops_size_one() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(4, 3, 64, 64, 16).launcher(&client);

    // One leading dim: right-aligns to B1 (the trailing axis of the full list).
    let one_batch = launch
        .arg::<f32>(binding(&client, &[3, 64, 16]))
        .subspace(&[M, K])
        .batches(&[B0, B1])
        .build();
    assert!(one_batch.space.contains(B1));
    assert!(!one_batch.space.contains(B0));

    // A size-1 dim drops out entirely (broadcast omission).
    let broadcast = launch
        .arg::<f32>(binding(&client, &[1, 64, 16]))
        .subspace(&[M, K])
        .batches(&[B0, B1])
        .build();
    assert!(!broadcast.space.contains(B0));
    assert!(!broadcast.space.contains(B1));
}

// ---- Launcher::vector_size -------------------------------------------------

#[test]
fn vector_size_picks_widest_qualifying_line() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    // Everything divides: N's leaf edge is 8, both inner extents are 64.
    let launch = batched_space(1, 1, 64, 64, 16).launcher(&client);
    let rhs = binding(&client, &[16, 64]);
    let out = binding(&client, &[64, 64]);

    let v = launch.vector_size(N, &[(&rhs, &[K, N]), (&out, &[M, N])], size_of::<f32>());
    // The gate passed, so the pick is the hardware's widest line fitting the leaf edge (8).
    let expected = client
        .io_optimized_vector_sizes(size_of::<f32>())
        .filter(|&v| 8 % v == 0)
        .max()
        .unwrap_or(1);
    assert_eq!(v, expected);
    assert_eq!(8 % v, 0);
    assert_eq!(64 % v, 0);
}

#[test]
fn vector_size_falls_back_to_scalar() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(1, 1, 64, 64, 16).launcher(&client);
    let out = binding(&client, &[64, 64]);

    // An overhanging operand (k = 18 vs leaf 4) stays scalar: its masked accesses report
    // their length in lines and would wrongly clip.
    let overhang = batched_space(1, 1, 64, 64, 18).launcher(&client);
    let rhs = binding(&client, &[18, 64]);
    assert_eq!(
        overhang.vector_size(N, &[(&rhs, &[K, N]), (&out, &[M, N])], size_of::<f32>()),
        1
    );

    // Col-major (innermost stride ≠ 1): lines wouldn't land on contiguous scalars.
    let mut col_major = binding(&client, &[16, 64]);
    col_major.strides = vec![1, 16].into();
    assert_eq!(
        launch.vector_size(
            N,
            &[(&col_major, &[K, N]), (&out, &[M, N])],
            size_of::<f32>()
        ),
        1
    );

    // An inner extent no width divides (63) blocks every line size.
    let odd = binding(&client, &[16, 63]);
    assert_eq!(
        launch.vector_size(N, &[(&odd, &[K, N])], size_of::<f32>()),
        1
    );
}

#[test]
#[should_panic(expected = "cannot be vectorized")]
fn arg_checked_and_vectorized_panics() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    // k = 18 overhangs its leaf, so the derived check is true: vectorizing must refuse.
    let launch = batched_space(1, 1, 64, 64, 18).launcher(&client);
    let _ = launch
        .arg::<f32>(binding(&client, &[64, 18]))
        .subspace(&[M, K])
        .vectorize(4)
        .build();
}

#[test]
#[should_panic(expected = "innermost dim")]
fn vector_size_axis_must_label_innermost() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(1, 1, 64, 64, 16).launcher(&client);
    let lhs = binding(&client, &[64, 16]);
    // lhs's innermost dim is K, not N: asking for N-lines over it is a labeling bug.
    let _ = launch.vector_size(N, &[(&lhs, &[M, K])], size_of::<f32>());
}

#[test]
#[should_panic(expected = "batch axes given")]
fn arg_more_batch_dims_than_axes_panics() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(4, 3, 64, 64, 16).launcher(&client);
    let _ = launch
        .arg::<f32>(binding(&client, &[4, 3, 64, 16]))
        .subspace(&[M, K])
        .batches(&[B1])
        .build();
}

// ---- StridedTileArgLaunch::quantized ---------------------------------------

/// Attach `scheme` to an `M×K` operand served in `v`-wide lines. Every rule below is also an
/// in-kernel assumption, so the launch is the one place a violation can still be seen: an
/// in-kernel assert fires on a device thread, which surfaces as zeroed output.
fn quantize(v: usize, scheme: QuantScheme) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let space = batched_space(1, 1, 64, 64, 16).project(&[M, K]);
    let _ = StridedTileArgLaunch::<i8, _>::strided(
        binding(&client, &[64, 16]).into_tensor_arg(),
        v,
        space.clone(),
        Storage::of(2, space.rank()),
    )
    .quantized(binding(&client, &[1, 8]).into_tensor_arg(), scheme);
}

fn quant_scheme(level: QuantLevel) -> QuantScheme {
    QuantScheme::default()
        .with_level(level)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S)
        .with_param(QuantParam::F32)
}

/// Scale blocks cut by the operand's own tiling: `K` is cut into 16-element tiles, so a 6-element
/// block leaves a tile origin mid-block, where the tile's window-relative lookup would silently
/// read a neighbour's scale.
#[test]
#[should_panic(expected = "straddle its 6-element scale blocks")]
fn quantized_block_straddling_a_cut_panics() {
    quantize(1, quant_scheme(QuantLevel::block([64, 6])));
}

/// 2-element blocks tile every `K` cut (16, then 4), so the tiling is fine — but a line is one
/// read, and a 4-wide line spans two of them.
#[test]
#[should_panic(expected = "straddles two scales")]
fn quantized_line_straddling_two_blocks_panics() {
    quantize(4, quant_scheme(QuantLevel::block([64, 2])));
}

/// Scales ride an `f32` buffer read straight through, so a narrower param would reinterpret its
/// bytes rather than convert them.
#[test]
#[should_panic(expected = "scales are read as f32")]
fn quantized_non_f32_param_panics() {
    quantize(
        1,
        quant_scheme(QuantLevel::Tensor).with_param(QuantParam::F16),
    );
}

/// A packed store's values are laid down along the innermost axis, so that is the only axis it may
/// pack on: the view unpacks a line's lanes into consecutive served values.
#[test]
#[should_panic(expected = "must pack along the innermost axis")]
fn quantized_packed_store_outer_axis_panics() {
    quantize(
        1,
        quant_scheme(QuantLevel::Tensor).with_store(QuantStore::PackedU32(2)),
    );
}

/// A served line must cover whole `u32`s: `Q8S` packs 4 values each, so a 1-wide line would ask
/// for a quarter of one.
#[test]
#[should_panic(expected = "packing factor")]
fn quantized_packed_store_narrow_line_panics() {
    quantize(
        1,
        quant_scheme(QuantLevel::Tensor).with_store(QuantStore::PackedU32(0)),
    );
}
