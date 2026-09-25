//! A stage whose block rows are swizzled ([`RowChunks::Swizzled`]) or padded
//! ([`RowChunks::Padded`]) holds the same product as one whose rows lie in order: filled whole,
//! filled through registers, and read by the manual mma transport — or, padded, by the cmma one —
//! a row-major rhs and a weight stored `{n, k}` alike.
//!
//! In `f16` summed in `f32`, at `16×16×16` where the device offers it and at the shape it does
//! otherwise (NVIDIA's manual mma is `16×8×16`); a device with no such instruction reports the
//! test skipped. Lines of
//! other byte widths (a 4-byte element's, a packed word's) are placed by the same rule, held on
//! the host by `RowPlacement`'s and `ChunkSwizzle`'s own tests.

use cubecl::{ir::ElemType, prelude::*, zspace::shape};
use cubek_test_utils::{HostData, HostDataType, TestInput, TestOutcome, ValidationResult};
use cubek_tile::*;

use super::matmul::Schedule;
use super::{Form, implied};

const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);

/// Which transport reads a stage's fragments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Leaf {
    /// cubek's manual mma, each lane computing its own addresses.
    Mma,
    /// The vendor's fragment API, rows off a pointer and a stride.
    Cmma,
}

/// `c = a · b` over a walk along `K` whose every region is one stage of `outer`'s tile, read a
/// fragment of `inner` at a time through `leaf`, the stage laid out as `storage`.
#[cube(launch)]
fn staged_k_walk<EI: Numeric, EA: Numeric, V: Size>(
    a: &TileArg<'_, EI, V>,
    b: &TileArg<'_, EI, V>,
    c: &TileArg<'_, EA, Const<1>>,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
    #[comptime] schedule: Schedule,
    #[comptime] leaf: Leaf,
    #[define(EI)] _input: ElemType,
    #[define(EA)] _sum: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = match comptime!(leaf) {
        Leaf::Mma => c.mma_accumulator::<EA, EI>(&a, comptime!(MmaIo::manual()), Monoid::Sum),
        Leaf::Cmma => c.cmma_accumulator::<EA, EI>(&a, Monoid::Sum),
    };
    acc.zero();
    let walk = space.over(&outer);
    let mut stages = Stages::smem(&walk, &a, &b, storage, depth);
    match comptime!(schedule) {
        Schedule::AheadInSlots => {
            stages.pipelined(walk, |slot, region| {
                let acc_o = acc.at(region);
                slot.consume(|a_s, b_s| {
                    for fragment in region.over(&inner).unrolled() {
                        let mut acc_f = acc_o.at(&fragment);
                        acc_f.mma(&a_s.at(&fragment), &b_s.at(&fragment), Semiring::SUM_PROD);
                    }
                });
            });
        }
        Schedule::ThroughRegisters => {
            stages.prefetched(walk, |slot, region| {
                let acc_o = acc.at(region);
                slot.consume(|a_s, b_s| {
                    for fragment in region.over(&inner).unrolled() {
                        let mut acc_f = acc_o.at(&fragment);
                        acc_f.mma(&a_s.at(&fragment), &b_s.at(&fragment), Semiring::SUM_PROD);
                    }
                });
            });
        }
    }
    for r0 in c.over(&outer).unrolled() {
        for r1 in r0.over(&inner).unrolled() {
            let mut c_w = c.at(&r1);
            c_w.copy_cast_from(&acc.at(&r1));
        }
    }
}

/// The `m × n × k` shape this device offers `leaf` at in `f16` summed in `f32`: `16×16×16` where
/// it does, else the first other it lists (NVIDIA's manual mma is `16×8×16`), so the stages are
/// read by the instruction each device has. Reported rather than silently passed where it has
/// none.
fn f16_shape(client: &cubecl::client::Client, leaf: Leaf) -> Option<(usize, usize, usize)> {
    let f16 = half::f16::elem_type_native();
    let f32 = f32::elem_type_native();
    let matmul = &client.properties().features.matmul;
    let configs = match leaf {
        Leaf::Mma => &matmul.mma,
        Leaf::Cmma => &matmul.cmma,
    };
    let shapes: Vec<(usize, usize, usize)> = configs
        .iter()
        .filter(|cfg| cfg.a_type == f16 && cfg.b_type == f16 && cfg.cd_type == f32)
        .map(|cfg| (cfg.m as usize, cfg.n as usize, cfg.k as usize))
        .collect();
    let shape = shapes
        .iter()
        .copied()
        .find(|&shape| shape == (16, 16, 16))
        .or_else(|| shapes.first().copied());
    if shape.is_none() {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device has no f16 {leaf:?} summed in f32"
        )))
        .enforce();
    }
    shape
}

/// Small integers, exact in `f16`, and differing between neighbours along every axis.
fn values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((i * 7 + seed) % 13) as f32 - 6.0)
        .collect()
}

/// One product through [`staged_k_walk`].
#[derive(Clone, Copy, Debug)]
struct Case {
    chunks: RowChunks,
    /// The problem's `M` and `N`, which one stage holds whole: past the block's 16, a stage holds
    /// several blocks along each.
    mn: usize,
    /// How much of `K` one stage holds.
    stage_k: usize,
    /// The width the operands are served in.
    v: usize,
    /// The rhs stored `{n, k}` rather than `{k, n}`.
    transposed: bool,
    /// Stages in flight.
    depth: usize,
    schedule: Schedule,
    leaf: Leaf,
}

impl Case {
    /// A 64-byte row (32 `f16` of `K`) in 8-wide lines, the rhs row-major, one stage, read by the
    /// manual mma transport.
    const DEFAULT: Self = Self {
        chunks: RowChunks::InOrder,
        mn: 16,
        stage_k: 32,
        v: 8,
        transposed: false,
        depth: 1,
        schedule: Schedule::AheadInSlots,
        leaf: Leaf::Mma,
    };
}

fn check(case: Case) {
    let Case {
        chunks,
        mn,
        stage_k,
        v,
        transposed,
        depth,
        schedule,
        leaf,
    } = case;
    let client = cubecl::test_device().client();
    let Some((edge_m, edge_n, edge_k)) = f16_shape(&client, leaf) else {
        return;
    };
    let (m, n, k) = (mn, mn, 128usize);
    // A row-major rhs's lines run along `N`, and a fragment narrower than a line starts inside one,
    // which the manual mma load refuses: reported, not run.
    if leaf == Leaf::Mma && !transposed && edge_n < v {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "a {v}-wide line runs past the {edge_n}-wide mma fragment along N"
        )))
        .enforce();
        return;
    }
    assert!(
        stage_k.is_multiple_of(edge_k),
        "{case:?}: a stage {stage_k} deep is not whole {edge_k}-deep steps"
    );
    let input = half::f16::elem_type_native();
    let a = values(m * k, 0);
    let b = values(k * n, 5);
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| a[i * k + p] * b[p * n + j]).sum()
        })
        .collect();
    // A `{n, k}` weight holds the same logical `b`, laid down along `k`.
    let b_stored: Vec<f32> = match transposed {
        false => b.clone(),
        true => (0..n * k).map(|idx| b[(idx % k) * n + idx / k]).collect(),
    };
    let (a_handle, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(input)
        .custom(a)
        .generate_with_f32_host_data();
    let b_shape = if transposed {
        shape![n, k]
    } else {
        shape![k, n]
    };
    let (b_handle, _) = TestInput::builder(client.clone(), b_shape)
        .dtype(input)
        .custom(b_stored)
        .generate_with_f32_host_data();
    let out = TestInput::builder(client.clone(), shape![m, n])
        .dtype(f32::elem_type_native())
        .zeros()
        .generate_without_host_data();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge_m), (N, edge_n), (K, edge_k)])
                .walk(&[(M, m / edge_m), (N, n / edge_n), (K, stage_k / edge_k)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    // A stage holds the whole of `M` and `N`, `stage_k` of `K`; a block is one fragment's rows,
    // `stage_k` deep, and the stage a grid of them wherever `mn` passes the fragment's edge.
    let storage = StageStorage::Tiled {
        block: vec![(M, edge_m), (N, edge_n), (K, stage_k)],
        chunks,
    };
    let b_axes: &'static [Axis] = if transposed { &[N, K] } else { &[K, N] };
    let bind = |binding, axes: &'static [Axis], width| {
        launcher.arg(binding).axes(axes).vectorize(width).build()
    };
    staged_k_walk::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        v,
        bind(a_handle.clone().binding(), &[M, K], v).arg(),
        bind(b_handle.clone().binding(), b_axes, v).arg(),
        bind(out.clone().binding(), &[M, N], 1).arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        storage,
        depth,
        schedule,
        leaf,
        input,
        f32::elem_type_native(),
    );

    let got = HostData::from_tensor_handle(&client, out, HostDataType::F32);
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    cubek_test_utils::assert_equals_approx(&got, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// A 64-byte row (32 `f16` of `K`), the block the cmma transport conflicts on, swizzled or padded:
/// the same product as in order.
#[test]
fn a_placed_stage_holds_the_product_of_one_in_order() {
    for chunks in [RowChunks::InOrder, RowChunks::Swizzled, RowChunks::Padded] {
        check(Case {
            chunks,
            ..Case::DEFAULT
        });
    }
}

/// A 32-byte row (16 `f16` of `K`), whose rows take two keys or pad by half their length.
#[test]
fn a_placed_stage_one_instruction_deep_holds_the_product() {
    for chunks in [RowChunks::Swizzled, RowChunks::Padded] {
        check(Case {
            chunks,
            stage_k: 16,
            ..Case::DEFAULT
        });
    }
}

/// A 128-byte row (64 `f16` of `K`), four instructions deep.
#[test]
fn a_placed_stage_four_instructions_deep_holds_the_product() {
    for chunks in [RowChunks::Swizzled, RowChunks::Padded] {
        for transposed in [false, true] {
            check(Case {
                chunks,
                stage_k: 64,
                transposed,
                ..Case::DEFAULT
            });
        }
    }
}

/// Lines narrower than a chunk: four lines move together, or four pad a row.
#[test]
fn a_placed_stage_of_narrow_lines_holds_the_product() {
    for chunks in [RowChunks::Swizzled, RowChunks::Padded] {
        check(Case {
            chunks,
            v: 2,
            ..Case::DEFAULT
        });
    }
}

/// A weight stored `{n, k}`, staged as it lies and read col-major, each lane's register vector a
/// run along its lines.
#[test]
fn a_weight_stored_along_k_is_read_where_it_lies() {
    for chunks in [RowChunks::InOrder, RowChunks::Swizzled, RowChunks::Padded] {
        check(Case {
            chunks,
            transposed: true,
            ..Case::DEFAULT
        });
    }
}

/// Two stages in flight, and the next held in registers across the contraction: both fills
/// write a placed stage where its reads look.
#[test]
fn a_placed_stage_is_filled_ahead_and_through_registers() {
    for chunks in [RowChunks::Swizzled, RowChunks::Padded] {
        for schedule in [Schedule::AheadInSlots, Schedule::ThroughRegisters] {
            for transposed in [false, true] {
                check(Case {
                    chunks,
                    transposed,
                    depth: 2,
                    schedule,
                    ..Case::DEFAULT
                });
            }
        }
    }
}

/// The cmma transport reads a padded stage off its pointer and its stride, the row pitched past
/// its padding: the same product as in order, a row-major rhs and a weight stored `{n, k}` alike.
#[test]
fn the_fragment_api_reads_a_padded_stage() {
    for chunks in [RowChunks::InOrder, RowChunks::Padded] {
        for transposed in [false, true] {
            check(Case {
                chunks,
                transposed,
                leaf: Leaf::Cmma,
                ..Case::DEFAULT
            });
        }
    }
}

/// A stage of two blocks along `M` and `N` each: a block's rows are pitched or swizzled within it,
/// and the next block starts past the last one's padding.
#[test]
fn a_placed_stage_of_several_blocks_holds_the_product() {
    for chunks in [RowChunks::Swizzled, RowChunks::Padded] {
        for transposed in [false, true] {
            for schedule in [Schedule::AheadInSlots, Schedule::ThroughRegisters] {
                check(Case {
                    chunks,
                    mn: 32,
                    transposed,
                    schedule,
                    ..Case::DEFAULT
                });
            }
        }
    }
}

/// Lines wider than a chunk (16 `f16`, 32 bytes): a swizzle moves whole lines, and a pad is one
/// line long.
#[test]
fn a_placed_stage_of_wide_lines_holds_the_product() {
    for chunks in [RowChunks::Swizzled, RowChunks::Padded] {
        for transposed in [false, true] {
            check(Case {
                chunks,
                v: 16,
                stage_k: 64,
                transposed,
                ..Case::DEFAULT
            });
        }
    }
}

/// The cmma transport over a stage of several padded blocks, whose rows it reads off a pointer
/// and the pitched stride.
#[test]
fn the_fragment_api_reads_a_padded_stage_of_several_blocks() {
    for transposed in [false, true] {
        check(Case {
            chunks: RowChunks::Padded,
            mn: 32,
            transposed,
            leaf: Leaf::Cmma,
            ..Case::DEFAULT
        });
    }
}
