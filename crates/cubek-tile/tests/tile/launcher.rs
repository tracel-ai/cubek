//! Unit tests for [`Launcher`]: geometry read off the concrete space, kernel space dynamic.

use cubecl::{
    TestRuntime,
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype},
};
use cubek_tile::{
    Axis, Boundary, Buffering, CubeAxis, Cut, DequantAt, Divisor, Offset, Operand, PhysicalAxisMap,
    Projection, Residence, Scale, StorageTiling, StridedOperand, TileSpec, Tiling, WalkOrder,
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

#[test]
fn launcher_over_frees_only_the_listed_axes() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(1, 1, 64, 64, 16).launcher_over(&client, &[M, K]);

    assert!(launch.space().is_dynamic(M));
    assert!(launch.space().is_dynamic(K));
    assert!(!launch.space().is_dynamic(N));
}

/// An axis the space does not have would be dropped silently, leaving the kernel specialized along
/// the axis the caller meant to free.
#[test]
#[should_panic(expected = "is not an axis of this space")]
fn launcher_over_unknown_axis_panics() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let _ = batched_space(1, 1, 64, 64, 16).launcher_over(&client, &[Axis(9)]);
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

/// An `f32` operand over `axes` staged `Smem` at the one level: the gathered-arg tests'
/// hand-built stages (their spaces are assembled outside `Tiling::over`).
fn smem_operand(axes: &[Axis]) -> Operand {
    let mut operand = Operand::new(axes, f32::elem_type_native());
    operand.stage(Residence::Smem);
    operand
}

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
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axes(&batches, Cut::cube(CubeAxis::Z, 1))
                .axis(M, Cut::cube(CubeAxis::X, 16))
                .axis(N, Cut::cube(CubeAxis::Y, 32))
                .axis(K, Cut::sequential(k))
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axes(&batches, Cut::sequential(1))
                .axis(M, Cut::plane(8))
                .axis(N, Cut::plane(8))
                .axis(K, Cut::sequential(4))
        })
        .build()
}

#[test]
fn arg_derives_check_from_subspace_overhang() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    // k = 18 overhangs its leaf (4); M and N divide everywhere.
    let launch = batched_space(1, 1, 64, 64, 18).launcher(&client);

    let touches_k = launch
        .arg(binding(&client, &[64, 18]))
        .subspace(&[M, K])
        .build();
    assert!(touches_k.spec.is_checked());
    // Only the axis that overhangs carries the mask: M divides, so its coordinate is settled and
    // masking it would cost a comparison on every access that can never fail.
    assert_eq!(
        touches_k.spec.boundaries.as_slice(),
        &[None, Some(Boundary::Zero)]
    );

    let avoids_k = launch
        .arg(binding(&client, &[64, 64]))
        .subspace(&[M, N])
        .build();
    assert!(!avoids_k.spec.is_checked());
    assert!(avoids_k.spec.boundaries.is_empty());

    // An explicit override still wins over the derivation.
    let forced = launch
        .arg(binding(&client, &[64, 18]))
        .subspace(&[M, K])
        .checked(false)
        .build();
    assert!(!forced.spec.is_checked());
    assert!(forced.spec.boundaries.is_empty());
}

/// A storage-tiled operand is addressed by fewer coordinates than its buffer has dims, and the
/// boundary list follows the coordinates: sized off the physical rank it would land its per-axis
/// modes on the grid fragments, which no [`Window`] ever reads.
#[test]
fn arg_sizes_boundaries_by_coordinate_rank_under_storage_tiling() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    // k = 18 overhangs its leaf (4), so K's coordinate is the one that needs the mode.
    let launch = batched_space(1, 1, 64, 64, 18).launcher(&client);

    // 2 coordinate axes (M, K), tiled into 4 physical buffer dims: 4*16 = 64 and 3*6 = 18.
    let tiled = launch
        .arg(binding(&client, &[4, 3, 16, 6]))
        .subspace(&[M, K])
        .tiling(StorageTiling::uniform(2, 1))
        .with_boundary(Some(Boundary::Clamp))
        .build();

    assert_eq!(tiled.spec.projection.physical_rank(), 4);
    assert_eq!(tiled.spec.projection.coordinate_rank(), 2);
    assert_eq!(
        tiled.spec.boundaries.as_slice(),
        &[None, Some(Boundary::Clamp)]
    );
}

#[test]
fn arg_right_aligns_batches_and_drops_size_one() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(4, 3, 64, 64, 16).launcher(&client);

    // One leading dim: right-aligns to B1 (the trailing axis of the full list).
    let one_batch = launch
        .arg(binding(&client, &[3, 64, 16]))
        .subspace(&[M, K])
        .batches(&[B0, B1])
        .build();
    assert!(one_batch.spec.axes().contains(&B1));
    assert!(!one_batch.spec.axes().contains(&B0));

    // A size-1 dim drops out entirely (broadcast omission).
    let broadcast = launch
        .arg(binding(&client, &[1, 64, 16]))
        .subspace(&[M, K])
        .batches(&[B0, B1])
        .build();
    assert!(!broadcast.spec.axes().contains(&B0));
    assert!(!broadcast.spec.axes().contains(&B1));
}

// ---- StridedTileSource::gathered -------------------------------------------

/// A convolution-shaped input over `batched_space`: output positions `M` at `stride` and taps `K`
/// at `dilation` share one physical dim, channels `N` ride the other. Three logical axes over a
/// rank-2 buffer, which no list of dim labels describes.
fn window(stride: usize, dilation: usize, offset: impl Into<Offset>) -> Projection {
    Projection::new(
        &[M, K, N],
        &[
            PhysicalAxisMap::affine_with_offset(&[(M, stride), (K, dilation)], offset),
            PhysicalAxisMap::of(N),
        ],
    )
}

/// The mapping reaches the spec as given: the buffer keeps its two dims and the tile spans the
/// three logical axes, in the projection's own order.
#[test]
fn arg_gathered_states_its_own_mapping() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(1, 1, 64, 64, 16).launcher_over(&client, &[N]);

    let input = launch
        .arg(binding(&client, &[79, 64]))
        .gathered(window(1, 1, 0))
        .build();

    assert_eq!(input.spec.axes(), &[M, K, N]);
    assert_eq!(input.spec.projection, window(1, 1, 0));
    assert_eq!(input.spec.projection.physical_rank(), 2);
    // Nothing overhangs and the origin cannot go negative, so the gather stays unchecked.
    assert!(input.spec.boundaries.is_empty());
}

/// An overhanging axis arms the check through the affine map just as it does through a label: the
/// tap axis `K` is one of the two the gathered dim is addressed by.
#[test]
fn arg_gathered_derives_check_from_overhang() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    // k = 18 overhangs its leaf (4).
    let launch = batched_space(1, 1, 64, 64, 18).launcher_over(&client, &[N]);

    let input = launch
        .arg(binding(&client, &[81, 64]))
        .gathered(window(1, 1, 0))
        .build();
    // The gathered coordinate takes the mask; N is its own coordinate and divides, so it is
    // settled whatever the gather does.
    assert_eq!(
        input.spec.boundaries.as_slice(),
        &[Some(Boundary::Zero), None]
    );

    // An explicit override still wins over the derivation.
    let forced = launch
        .arg(binding(&client, &[81, 64]))
        .gathered(window(1, 1, 0))
        .checked(false)
        .build();
    assert!(forced.spec.boundaries.is_empty());
}

/// A padded window reads before the buffer's start whatever the tiling divides, so the derivation
/// arms on the offset's sign alone. A forward shift cannot underflow and stays unchecked.
#[test]
fn arg_gathered_derives_check_from_underflow() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(1, 1, 64, 64, 16).launcher_over(&client, &[N]);

    let padded = launch
        .arg(binding(&client, &[64, 64]))
        .gathered(window(1, 1, -1))
        .build();
    assert_eq!(
        padded.spec.boundaries.as_slice(),
        &[Some(Boundary::Zero), None]
    );

    // A runtime offset's sign is unknown at launch, so it arms the guard conservatively.
    let dynamic = launch
        .arg(binding(&client, &[64, 64]))
        .gathered(window(1, 1, Offset::Dynamic))
        .build();
    assert_eq!(
        dynamic.spec.boundaries.as_slice(),
        &[Some(Boundary::Zero), None]
    );

    let shifted = launch
        .arg(binding(&client, &[96, 64]))
        .gathered(window(1, 1, 1))
        .build();
    assert!(shifted.spec.boundaries.is_empty());
}

/// The stated mapping replaces the labeling whole, so a leftover `subspace` describes nothing and
/// is refused rather than silently dropped.
#[test]
#[should_panic(expected = "nothing left to describe")]
fn arg_gathered_alongside_a_subspace_panics() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(1, 1, 64, 64, 16).launcher_over(&client, &[N]);
    let _ = launch
        .arg(binding(&client, &[79, 64]))
        .subspace(&[M, N])
        .gathered(window(1, 1, 0))
        .build();
}

/// One map per buffer dim: a mapping that addresses fewer dims than the binding has would read
/// every coarser stride as if it were the operand's own.
#[test]
#[should_panic(expected = "addresses 2 dims but the binding has 3")]
fn arg_gathered_rank_mismatch_panics() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(1, 1, 64, 64, 16).launcher_over(&client, &[N]);
    let _ = launch
        .arg(binding(&client, &[4, 79, 64]))
        .gathered(window(1, 1, 0))
        .build();
}

/// The gather contract runs on the caller's thread now that the builder knows the served width:
/// the innermost dim is addressed in lines, so it must be one logical axis at coefficient 1.
#[test]
#[should_panic(expected = "innermost physical axis")]
fn arg_gathered_validates_the_innermost_dim() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(1, 1, 64, 64, 16).launcher_over(&client, &[M]);
    let _ = launch
        .arg(binding(&client, &[64, 79]))
        .gathered(Projection::new(
            &[M, K, N],
            &[
                PhysicalAxisMap::of(M),
                PhysicalAxisMap::affine(&[(N, 1), (K, 2)]),
            ],
        ))
        .vectorize(4)
        .checked(false)
        .build();
}

/// An axis sharing its dim with another has no extent of its own to read back here, but the
/// operand is free to ride it [`Dynamic`] anyway: whoever maps it identically states its size when
/// the op walks it. The builder sees one operand, so it is not the place to rule on that; an axis
/// no operand answers for is reported by `witnessed_space`, at expansion.
#[test]
fn arg_gathered_dynamic_axis_is_accepted() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    // `K` shares the gathered dim with `M`, so neither reads an extent off *this* operand.
    let launch = batched_space(1, 1, 64, 64, 16).launcher_over(&client, &[N, K]);
    let _ = launch
        .arg(binding(&client, &[79, 64]))
        .gathered(window(1, 1, 0))
        .build();
}

/// The axis a gather does identity-map still reads its own extent, so it is free to stay runtime
/// alongside the two that share a dim.
#[test]
fn arg_gathered_identity_axis_may_stay_dynamic() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(1, 1, 64, 64, 16).launcher_over(&client, &[N]);

    let input = launch
        .arg(binding(&client, &[79, 64]))
        .gathered(window(1, 1, 0))
        .build();
    assert_eq!(input.spec.axes(), &[M, K, N]);
    assert!(launch.space().is_dynamic(N));
    assert!(!launch.space().is_dynamic(M));
}

/// A runtime coefficient sizes its compacted window by its declared `max`, so the smem it stages
/// into holds every window the launch can then ask for.
#[test]
fn arg_gathered_dynamic_coefficient_stages_to_its_bound() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let staged = Tiling::new()
        .extents(&[(M, 64), (N, 64), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::cube(CubeAxis::X, 16))
                .axis(N, Cut::cube(CubeAxis::Y, 32))
                .axis(K, Cut::sequential(16))
        })
        .build()
        .launcher_over(&client, &[N]);
    let _ = staged
        .arg(binding(&client, &[79, 64]))
        .operand(&smem_operand(&[M, K, N]))
        .gathered(Projection::new(
            &[M, K, N],
            &[
                PhysicalAxisMap::scaled(&[
                    (M, Scale::Dynamic { max: 2 }),
                    (K, Scale::Dynamic { max: 3 }),
                ]),
                PhysicalAxisMap::of(N),
            ],
        ))
        .build();
}

/// A static rational projection stages uncompacted into shared memory.
#[test]
fn arg_gathered_rational_stages() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let staged = Tiling::new()
        .extents(&[(M, 64), (N, 64), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::cube(CubeAxis::X, 16))
                .axis(N, Cut::cube(CubeAxis::Y, 32))
                .axis(K, Cut::sequential(16))
        })
        .build()
        .launcher_over(&client, &[N]);
    let _ = staged
        .arg(binding(&client, &[79, 64]))
        .operand(&smem_operand(&[M, K, N]))
        .gathered(Projection::new(
            &[M, K, N],
            &[
                PhysicalAxisMap::affine(&[(M, 3), (K, 4)]).over(4),
                PhysicalAxisMap::of(N),
            ],
        ))
        .build();
}

/// A dynamic divisor stages against its `min`, the smallest divisor and so the widest window any
/// launch can ask for.
#[test]
fn arg_gathered_dynamic_divisor_stages_to_its_bound() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let staged = Tiling::new()
        .extents(&[(M, 64), (N, 64), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::cube(CubeAxis::X, 16))
                .axis(N, Cut::cube(CubeAxis::Y, 32))
                .axis(K, Cut::sequential(16))
        })
        .build()
        .launcher_over(&client, &[N]);
    let _ = staged
        .arg(binding(&client, &[79, 64]))
        .operand(&smem_operand(&[M, K, N]))
        .gathered(Projection::new(
            &[M, K, N],
            &[
                PhysicalAxisMap::affine(&[(M, 3), (K, 4)]).over(Divisor::Dynamic { min: 4 }),
                PhysicalAxisMap::of(N),
            ],
        ))
        .build();
}

/// The same shape with a divisor its coefficients cancel: `⌊(8m + 4k)/4⌋` steps like `2m + k`, so
/// `over` reduces it away and what reaches the stage is a plain strided gather, which compacts.
#[test]
fn arg_gathered_cancelling_divisor_stages() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let staged = Tiling::new()
        .extents(&[(M, 64), (N, 64), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::cube(CubeAxis::X, 16))
                .axis(N, Cut::cube(CubeAxis::Y, 32))
                .axis(K, Cut::sequential(16))
        })
        .build()
        .launcher_over(&client, &[N]);
    let projection = Projection::new(
        &[M, K, N],
        &[
            PhysicalAxisMap::affine(&[(M, 8), (K, 4)]).over(4),
            PhysicalAxisMap::of(N),
        ],
    );
    assert!(!projection.is_rational());
    let _ = staged
        .arg(binding(&client, &[512, 64]))
        .operand(&smem_operand(&[M, K, N]))
        .gathered(projection)
        .build();
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
#[should_panic(expected = "not provably in bounds")]
fn arg_checked_and_vectorized_panics() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    // k = 18 overhangs its leaf, so the derived check is true: vectorizing must refuse.
    let launch = batched_space(1, 1, 64, 64, 18).launcher(&client);
    let _ = launch
        .arg(binding(&client, &[64, 18]))
        .subspace(&[M, K])
        .vectorize(4)
        .build();
}

#[test]
fn arg_vectorized_with_outer_axis_overhang_succeeds() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    // M = 63 overhangs its leaf (8); N = 64 divides cleanly.
    let launch = batched_space(1, 1, 63, 64, 16).launcher(&client);
    let arg = launch
        .arg(binding(&client, &[63, 64]))
        .subspace(&[M, N])
        .vectorize(4)
        .build();
    assert!(arg.spec.is_checked());
    // M takes the mask, N carries the lines and needs none.
    assert_eq!(
        arg.spec.boundaries.as_slice(),
        &[Some(Boundary::Zero), None]
    );
}

/// `checked(true)` states the mode, not the axis list: the per-axis derivation still keeps it off
/// the axes that cannot leave the buffer. Only `checked(false)` disarms outright.
#[test]
fn arg_explicit_check_still_narrows_to_the_unsettled_axes() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    // k = 18 overhangs its leaf (4); M divides, so no override can put a mask on it.
    let launch = batched_space(1, 1, 64, 64, 18).launcher(&client);

    let forced = launch
        .arg(binding(&client, &[64, 18]))
        .subspace(&[M, K])
        .checked(true)
        .build();
    assert_eq!(
        forced.spec.boundaries.as_slice(),
        &[None, Some(Boundary::Zero)]
    );
}

/// The list is shaped over coordinate axes, and the setter is where a hand-built spec should learn
/// that: `Tile::of` would only catch it in-kernel, one comptime expansion away from the call.
#[test]
#[should_panic(expected = "2 coordinate axes")]
fn tile_spec_boundaries_must_match_the_coordinate_rank() {
    let _ = TileSpec::new(Projection::direct(&[M, N])).boundaries(&[None]);
}

#[test]
fn arg_gathered_clamp_vectorized_exemption() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let launch = batched_space(1, 1, 64, 64, 16).launcher_over(&client, &[N]);
    // 3 logical axes [M, K, N] mapped over 2 coordinate axes:
    // Spatial (M, K) on coordinate 0 (clamped), channel N on coordinate 1 (vectorized & in-bounds).
    let input = launch
        .arg(binding(&client, &[79, 64]))
        .gathered(window(1, 1, 0))
        .vectorize(4)
        .with_boundary(Some(Boundary::Clamp))
        .build();

    assert_eq!(
        input.spec.boundaries.as_slice(),
        &[Some(Boundary::Clamp), None]
    );
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
        .arg(binding(&client, &[4, 3, 64, 16]))
        .subspace(&[M, K])
        .batches(&[B1])
        .build();
}

// ---- StridedTileSource::quantized ------------------------------------------

/// Attach `scheme` to an `M×K` operand served in `v`-wide lines. Every rule below is also an
/// in-kernel assumption, so the launch is the one place a violation can still be seen: an
/// in-kernel assert fires on a device thread, which surfaces as zeroed output.
fn quantize(v: usize, scheme: QuantScheme) {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let space = batched_space(1, 1, 64, 64, 16).project(&[M, K]);
    let _ = StridedOperand::source(binding(&client, &[64, 16]))
        .space(&space)
        .subspace(&[M, K])
        .vectorize(v)
        .checked(false)
        .quantized(&[binding(&client, &[1, 8])], scheme, DequantAt::Read)
        .build();
}

fn quant_scheme() -> QuantScheme {
    QuantScheme::default()
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S)
}

/// Scale blocks cut by the operand's own tiling: `K` is cut into 16-element tiles, so a 6-element
/// block leaves a tile origin mid-block, where the tile's window-relative lookup would silently
/// read a neighbour's scale.
#[test]
#[should_panic(expected = "straddle its 6-element scale blocks")]
fn quantized_block_straddling_a_cut_panics() {
    quantize(1, quant_scheme().per_block([64, 6], ScaleDtype::F32));
}

/// 2-element blocks tile every `K` cut (16, then 4), so the tiling is fine — but a line is one
/// read, and a 4-wide line spans two of them.
#[test]
#[should_panic(expected = "straddles two scales")]
fn quantized_line_straddling_two_blocks_panics() {
    quantize(4, quant_scheme().per_block([64, 2], ScaleDtype::F32));
}

/// Scales ride an `f32` buffer read straight through, so a narrower param would reinterpret its
/// bytes rather than convert them.
#[test]
#[should_panic(expected = "scales are read as f32")]
fn quantized_non_f32_param_panics() {
    quantize(1, quant_scheme().per_tensor(ScaleDtype::F16));
}

/// A packed store's values are laid down along the innermost axis, so that is the only axis it may
/// pack on: the view unpacks a line's lanes into consecutive served values.
#[test]
#[should_panic(expected = "must pack along the innermost axis")]
fn quantized_packed_store_outer_axis_panics() {
    quantize(
        1,
        quant_scheme()
            .per_tensor(ScaleDtype::F32)
            .with_store(QuantStore::PackedU32(2)),
    );
}

/// A served line must cover whole `u32`s: `Q8S` packs 4 values each, so a 1-wide line would ask
/// for a quarter of one.
#[test]
#[should_panic(expected = "packing factor")]
fn quantized_packed_store_narrow_line_panics() {
    quantize(
        1,
        quant_scheme()
            .per_tensor(ScaleDtype::F32)
            .with_store(QuantStore::PackedU32(0)),
    );
}
