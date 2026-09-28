//! Matmul as a client of the axis-agnostic tile DSL engine: every kernel here writes its own
//! walk, allocates its own stages and opens its own accumulator, one kernel per shape the tests
//! drive.
#![allow(non_snake_case)]

use cubecl::{
    cmma::{MatrixIdent, MatrixLayout},
    features::TypeUsage,
    ir::ElemType,
    prelude::*,
    std::tensor::TensorHandle,
    zspace::shape,
};
use cubek_quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
use cubek_test_utils::{
    HostData, HostDataType, TestInput, TestOutcome, TileInput, ValidationResult,
    assert_equals_approx, skip_unless_plane_holds,
};

use cubek_tile::*;

use half::f16;

use super::references;
use super::{Form, implied, uncut};

/// Skip guard for the tensor-core tests in this file, which all hardcode `8x8x8` `f32` fragments
/// (the native Metal simdgroup shape). Drivers accept only the exact fragment shapes they
/// advertise, so *some* config is not enough; returns `false` (after enforcing a skip) if absent.
pub(crate) fn require_cmma_8x8x8_f32(client: &Client) -> bool {
    let f32_ty = f32::elem_type_native();
    let supported = client.properties().features.matmul.cmma.iter().any(|cfg| {
        cfg.a_type == f32_ty
            && cfg.b_type == f32_ty
            && cfg.cd_type == f32_ty
            && cfg.m == 8
            && cfg.n == 8
            && cfg.k == 8
    });
    if !supported {
        TestOutcome::Validated(ValidationResult::Skipped(
            "device has no 8x8x8 f32 cmma (tensor-core) fragment support".to_string(),
        ))
        .enforce();
    }
    supported
}

/// The manual-mma twin of [`require_cmma_8x8x8_f32`]. The *shape*, not just the feature: a
/// backend can advertise manual mma and offer only `16x16x16` (gfx1151 does), and running `8x8x8`
/// there is an instruction the hardware lacks: it reads back zeros, which looks like a leaf bug.
fn require_mma_8x8x8_f32(client: &Client) -> bool {
    let f32_ty = f32::elem_type_native();
    let supported = client.properties().features.matmul.mma.iter().any(|cfg| {
        cfg.a_type == f32_ty
            && cfg.b_type == f32_ty
            && cfg.cd_type == f32_ty
            && (cfg.m, cfg.n, cfg.k) == (8, 8, 8)
    });
    if !supported {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend offers no 8x8x8 f32 manual mma".to_string(),
        ))
        .enforce();
    }
    supported
}

fn require_native_i8(client: &Client) -> bool {
    let supported = i8::supported_uses(client).contains(TypeUsage::Conversion);
    if !supported {
        TestOutcome::Validated(ValidationResult::Skipped(
            "backend has no native i8".to_string(),
        ))
        .enforce();
    }
    supported
}

// Matmul's axes: the labels this client gives the engine's opaque `Axis`. `B`
// is the leading batch axis; `M`/`N`/`K` are the matrix axes.
const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
const B: Axis = Axis(3);

// A broadcast batch carried as two independent axes: `lhs` spans `B0`, `rhs` spans
// `B1`, the output spans both. Each operand simply omits the axis it broadcasts.
const B0: Axis = Axis(4);
const B1: Axis = Axis(5);

// A second contracted axis, so a contraction that is otherwise a plain matmul takes the N-D nest.
const K2: Axis = Axis(6);

/// The software instruction most tests contract through: a 16-cell budget, no edge split, no
/// unit fan-out.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// Where an operand is read from in the kernels that offer both: staged in shared memory, or
/// where it lies in global memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Serve {
    Staged,
    Direct,
}

/// The kernel-side level of one cube naming nothing: the whole space is its box.
fn one_cube() -> Level {
    Levels::leaf(&[]).cubes(&[]).level()
}

/// The edge of the innermost level naming `axis`: the tile the inputs are laid out in.
fn leaf_edge(levels: &[Level], axis: Axis) -> usize {
    levels
        .iter()
        .rev()
        .find_map(|level| level.tile(axis))
        .unwrap_or_else(|| panic!("leaf_edge: no level names {axis:?}"))
}

/// The walk between a worker level and the steps under it, where a tiling states one: the
/// worker's run of boxes. Two levels have none; three have it in the middle.
fn runs_and_steps(launcher: &Launcher) -> (Option<Level>, Level) {
    match launcher.partitioning().levels().len() {
        2 => (None, launcher.partitioning().level(1)),
        3 => (
            Some(launcher.partitioning().level(1)),
            launcher.partitioning().level(2),
        ),
        depth => panic!("runs_and_steps: no kernel walks {depth} levels"),
    }
}

/// `A·B` off row-major `arange` operands: `lhs(i, p) = i·k + p`, `rhs(p, j) = p·n + j`.
fn arange_matmul_reference(m: usize, n: usize, k: usize) -> Vec<f32> {
    (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| ((i * k + p) * (p * n + j)) as f32).sum()
        })
        .collect()
}

/// `c == a·b` for the two `arange` operands most tests share.
fn assert_matmul_arange(client: &Client, handle: TensorHandle, m: usize, n: usize, k: usize) {
    let output = HostData::from_tensor_handle(client, handle, HostDataType::F32);
    let (_, expected) = TestInput::builder(client.clone(), shape![m, n])
        .custom(arange_matmul_reference(m, n, k))
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

// ---- the kernels ------------------------------------------------------------------

/// `c = a · b` with every operand read where it lies: `outer`'s workers each take their box of
/// it (through `runs` of boxes where the tiling states that level) and the leaf runs the software
/// instruction under `config` on each `inner` region, folding under `semiring` from its identity.
#[cube(launch)]
fn matmul_in_place<E: Numeric, AV: Size, BV: Size, CV: Size>(
    a: &TileArg<'_, E, AV>,
    b: &TileArg<'_, E, BV>,
    c: &TileArg<'_, E, CV>,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] runs: Option<Level>,
    #[comptime] inner: Level,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    for outer in space.over(&outer) {
        match comptime!(runs.clone()) {
            Some(runs) => {
                for run in outer.over(&runs) {
                    contract_in_place::<E>(
                        &a,
                        &b,
                        &c,
                        &run,
                        comptime!(inner.clone()),
                        config,
                        semiring,
                    );
                }
            }
            None => contract_in_place::<E>(
                &a,
                &b,
                &c,
                &outer,
                comptime!(inner.clone()),
                config,
                semiring,
            ),
        }
    }
}

/// One worker's box of `c = a · b`: every window of `c` initialized once, then each region of
/// `inner` contracted where the operands lie.
#[cube]
fn contract_in_place<E: Numeric>(
    a: &Tile<E>,
    b: &Tile<E>,
    c: &Tile<E>,
    owned: &Region,
    #[comptime] inner: Level,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
) {
    // This instance's windows of `c`, each initialized once: the level projected onto `c`'s
    // own axes walks nothing it does not span.
    let c_o = c.at(owned);
    for region in c_o.over(&inner) {
        let mut c_w = c_o.at(&region);
        c_w.init(Monoid::identity::<E>(comptime!(semiring.add())));
    }
    for region in owned.over(&inner) {
        let mut c_r = c.at(&region);
        c_r.mma_with(&a.at(&region), &b.at(&region), config, semiring);
    }
}

/// `c = a · b`, each cube's box of it, through `runs` of boxes where the tiling states that
/// level: both operands staged in shared memory per region of the walk under the cube, `depth`
/// regions in flight through the stages.
#[cube(launch)]
fn matmul_smem_ring<E: Numeric, V: Size>(
    a: &TileArg<'_, E, V>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    space: Partitioning,
    #[comptime] cubes: Level,
    #[comptime] runs: Option<Level>,
    #[comptime] steps: Level,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    for cube in space.over(&cubes) {
        match comptime!(runs.clone()) {
            Some(runs) => {
                for run in cube.over(&runs) {
                    contract_staged::<E>(&a, &b, &c, &run, comptime!(steps.clone()), depth);
                }
            }
            None => contract_staged::<E>(&a, &b, &c, &cube, comptime!(steps.clone()), depth),
        }
    }
}

/// One cube's box of `c = a · b`, zeroed once, then contracted through stages staging each
/// region of `steps`.
#[cube]
fn contract_staged<E: Numeric>(
    a: &Tile<E>,
    b: &Tile<E>,
    c: &Tile<E>,
    owned: &Region,
    #[comptime] steps: Level,
    #[comptime] depth: usize,
) {
    let a = a.at(owned);
    let b = b.at(owned);
    let c = c.at(owned);
    for region in c.over(&steps) {
        let mut c_w = c.at(&region);
        c_w.zero();
    }
    let walk = owned.over(&steps);
    let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, depth);
    stages.pipelined(walk, |slot, region| {
        let mut c_r = c.at(region);
        slot.consume(|a_s, b_s| {
            c_r.mma_with(a_s, b_s, REGISTER_BLOCK, Semiring::SUM_PROD);
        });
    });
}

/// Which of the ring's schedules a staged test kernel drives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Schedule {
    /// [`pipelined`]: the next region's slot filled ahead of the contraction.
    AheadInSlots,
    /// [`pipelined_through_registers`]: the next region read into registers across the
    /// contraction and written after it.
    ThroughRegisters,
}

/// [`matmul_smem_ring`] under either of the ring's schedules.
#[cube(launch)]
fn matmul_smem_ring_scheduled<E: Numeric, V: Size>(
    a: &TileArg<'_, E, V>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    space: Partitioning,
    #[comptime] cubes: Level,
    #[comptime] steps: Level,
    #[comptime] depth: usize,
    #[comptime] schedule: Schedule,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    for cube in space.over(&cubes) {
        let a = a.at(&cube);
        let b = b.at(&cube);
        let c = c.at(&cube);
        for region in c.over(&steps) {
            let mut c_w = c.at(&region);
            c_w.zero();
        }
        let walk = cube.over(&steps);
        let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, depth);
        match comptime!(schedule) {
            Schedule::AheadInSlots => {
                stages.pipelined(walk, |slot, region| {
                    let c_r = c.at(region);
                    slot.consume(|a_s, b_s| {
                        contract_on_the_last_unit::<E>(&c_r, a_s, b_s);
                    });
                });
            }
            Schedule::ThroughRegisters => {
                stages.prefetched(walk, |slot, region| {
                    let c_r = c.at(region);
                    slot.consume(|a_s, b_s| {
                        contract_on_the_last_unit::<E>(&c_r, a_s, b_s);
                    });
                });
            }
        }
    }
}

/// `c += a · b` on the cube's last unit alone, the rest having only filled the slot: the
/// contraction adds into `c` where it lies, so one unit makes it, and the last so that most of the
/// units that wrote the slot sit in other planes, where only a barrier orders their writes before
/// this read.
#[cube]
fn contract_on_the_last_unit<E: Numeric>(c: &Tile<E>, a: &Tile<E>, b: &Tile<E>) {
    if UNIT_POS == CUBE_DIM - 1 {
        let mut c = c.clone();
        c.mma_with(a, b, REGISTER_BLOCK, Semiring::SUM_PROD);
    }
}

/// [`matmul_smem_ring`] walking its regions last to first.
#[cube(launch)]
fn matmul_smem_ring_reversed<E: Numeric, V: Size>(
    a: &TileArg<'_, E, V>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();
    let walk = space.over(&level).reversed();
    let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, depth);
    stages.pipelined(walk, |slot, region| {
        let mut c_r = c.at(region);
        slot.consume(|a_s, b_s| {
            c_r.mma_with(a_s, b_s, REGISTER_BLOCK, Semiring::SUM_PROD);
        });
    });
}

/// `c += a · b`: [`matmul_smem_ring`] folding onto what `c` holds, for the caller that owns the
/// init.
#[cube(launch)]
fn matmul_smem_ring_accumulate<E: Numeric, V: Size>(
    a: &TileArg<'_, E, V>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let walk = space.over(&level);
    let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, depth);
    stages.pipelined(walk, |slot, region| {
        let mut c_r = c.at(region);
        slot.consume(|a_s, b_s| {
            c_r.mma_with(a_s, b_s, REGISTER_BLOCK, Semiring::SUM_PROD);
        });
    });
}

/// `c = a · b` with the lhs alone staged, the rhs read where it lies, `depth` regions in flight.
#[cube(launch)]
fn matmul_lhs_smem_ring<E: Numeric, V: Size>(
    a: &TileArg<'_, E, V>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();
    let walk = space.over(&level);
    let mut stages = Stages::smem_single(&walk, &a, StageStorage::Strided, depth);
    stages.pipelined(walk, |slot, region| {
        let mut c_r = c.at(region);
        let b_r = b.at(region);
        slot.consume(|a_s| {
            c_r.mma_with(a_s, &b_r, REGISTER_BLOCK, Semiring::SUM_PROD);
        });
    });
}

/// `c = a · b` with the scalar rhs staged into `width`-wide shared-memory lines: the stage pads
/// its innermost axis out to whole lines an unvectorized global read could not give.
#[cube(launch)]
fn matmul_padded_rhs_stage<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] width: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    // This instance's windows of `c`, each initialized once: the level projected
    // onto `c`'s own axes walks nothing it does not span.
    for region in c.over(&level) {
        let mut c_w = c.at(&region);
        c_w.zero();
    }
    let walk = space.over(&level);
    let mut stages = Stages::smem_single_at(
        &walk,
        &b,
        StageStorage::Strided,
        comptime!(Some(width)),
        1usize,
    );
    stages.pipelined(walk, |slot, region| {
        let mut c_r = c.at(region);
        let a_r = a.at(region);
        slot.consume(|b_s| {
            c_r.mma_with(&a_r, b_s, REGISTER_BLOCK, Semiring::SUM_PROD);
        });
    });
}

/// [`matmul_padded_rhs_stage`]'s lhs twin, the padded stage one level down: the outer level reads
/// the lhs where it lies, the inner one stages it `width` wide.
#[cube(launch)]
fn matmul_padded_lhs_stage_two_levels<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
    #[comptime] width: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    // This instance's windows of `c`, each initialized once: the level projected
    // onto `c`'s own axes walks nothing it does not span.
    for region in c.over(&outer) {
        let mut c_w = c.at(&region);
        c_w.zero();
    }
    for outer in space.over(&outer) {
        let c_o = c.at(&outer);
        let a_o = a.at(&outer);
        let b_o = b.at(&outer);
        let walk = outer.over(&inner);
        let mut stages = Stages::smem_single_at(
            &walk,
            &a_o,
            StageStorage::Strided,
            comptime!(Some(width)),
            1usize,
        );
        stages.pipelined(walk, |slot, region| {
            let mut c_r = c_o.at(region);
            let b_r = b_o.at(region);
            slot.consume(|a_s| {
                c_r.mma_with(a_s, &b_r, REGISTER_BLOCK, Semiring::SUM_PROD);
            });
        });
    }
}

/// Two levels: the outer stages both operands (laid out as `storage`, `depth` regions in flight),
/// the inner walks the stage's final tiles where they lie.
#[cube(launch)]
fn matmul_two_levels_smem_then_in_place<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();
    let walk = space.over(&outer);
    let mut stages = Stages::smem(&walk, &a, &b, storage, depth);
    stages.pipelined(walk, |slot, region| {
        let c_o = c.at(region);
        slot.consume(|a_s, b_s| {
            for cell in region.over(&inner) {
                let mut c_r = c_o.at(&cell);
                c_r.mma_with(
                    &a_s.at(&cell),
                    &b_s.at(&cell),
                    REGISTER_BLOCK,
                    Semiring::SUM_PROD,
                );
            }
        });
    });
}

/// [`matmul_two_levels_smem_then_in_place`] with the inner walk last to first.
#[cube(launch)]
fn matmul_two_levels_smem_then_in_place_reversed<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();
    let walk = space.over(&outer);
    let mut stages = Stages::smem(&walk, &a, &b, storage, depth);
    stages.pipelined(walk, |slot, region| {
        let c_o = c.at(region);
        slot.consume(|a_s, b_s| {
            for cell in region.over(&inner).reversed() {
                let mut c_r = c_o.at(&cell);
                c_r.mma_with(
                    &a_s.at(&cell),
                    &b_s.at(&cell),
                    REGISTER_BLOCK,
                    Semiring::SUM_PROD,
                );
            }
        });
    });
}

/// Two levels, both staging: the inner stages restages each final tile out of the outer stage.
#[cube(launch)]
fn matmul_two_levels_smem_then_smem<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
    #[comptime] storage: StageStorage,
    #[comptime] depth_outer: usize,
    #[comptime] depth_inner: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    // This instance's windows of `c`, each initialized once: the level projected
    // onto `c`'s own axes walks nothing it does not span.
    for region in c.over(&outer) {
        let mut c_w = c.at(&region);
        c_w.zero();
    }
    let walk = space.over(&outer);
    let mut stages = Stages::smem(&walk, &a, &b, comptime!(storage.clone()), depth_outer);
    stages.pipelined(walk, |slot, region| {
        let c_o = c.at(region);
        slot.consume(|a_s, b_s| {
            let cells = region.over(&inner);
            let mut inner_ring =
                Stages::smem(&cells, a_s, b_s, comptime!(storage.clone()), depth_inner);
            inner_ring.pipelined(cells, |slot, cell| {
                let mut c_r = c_o.at(cell);
                slot.consume(|a_i, b_i| {
                    c_r.mma_with(a_i, b_i, REGISTER_BLOCK, Semiring::SUM_PROD);
                });
            });
        });
    });
}

/// `c = a · b` through a register block promoted out of memory: opened over the one level,
/// seeded with the semiring's identity, every region's contraction folded into it, cast back down
/// on drain.
#[cube(launch)]
fn promoted_matmul_in_place<E: Numeric, EA: Numeric, AV: Size, BV: Size, CV: Size>(
    a: &TileArg<'_, E, AV>,
    b: &TileArg<'_, E, BV>,
    c: &TileArg<'_, E, CV>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
    #[define(E)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.block_accumulator::<EA, E, E>(&a, &b, config, comptime!(semiring.add()));
    acc.init(Monoid::identity::<EA>(comptime!(semiring.add())));
    for region in space.over(&level) {
        let mut acc_r = acc.at(&region);
        acc_r.mma(&a.at(&region), &b.at(&region), semiring);
    }
    for r0 in c.over(&level).unrolled() {
        let mut c_w = c.at(&r0);
        c_w.copy_cast_from(&acc.at(&r0));
    }
}

/// [`promoted_matmul_in_place`] over the two-level cube/plane nest a real gemm composes: the
/// block is opened per plane region and contracted along the inner K walk.
#[cube(launch)]
fn promoted_matmul_two_levels_in_place<E: Numeric, EA: Numeric, V: Size>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    space: Partitioning,
    #[comptime] config: RegisterBlock,
    #[define(E)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    for cube in space {
        for plane in cube {
            let c_p = c.at(&plane);
            let a_p = a.at(&plane);
            let b_p = b.at(&plane);
            let mut acc = c_p.block_accumulator::<EA, E, E>(&a_p, &b_p, config, Monoid::Sum);
            acc.zero();
            for step in plane {
                let mut acc_s = acc.at(&step);
                acc_s.mma(&a_p.at(&step), &b_p.at(&step), Semiring::SUM_PROD);
            }
            for r0 in c_p.walk().unrolled() {
                let mut c_p_w = c_p.at(&r0);
                c_p_w.copy_cast_from(&acc.at(&r0));
            }
        }
    }
}

/// A register block opened above two levels, the inner one staging both operands and *cutting*
/// the block: each inner region selects its own fragment of the partition, so that walk unrolls
/// and hands every region comptime coordinates.
#[cube(launch)]
fn block_matmul_two_levels_smem_below<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.block_accumulator::<E, E, E>(&a, &b, REGISTER_BLOCK, Monoid::Sum);
    acc.zero();
    for outer in space.over(&outer) {
        let acc_o = acc.at(&outer);
        let a_o = a.at(&outer);
        let b_o = b.at(&outer);
        let walk = outer.over(&inner).unrolled();
        let mut stages = Stages::smem(&walk, &a_o, &b_o, StageStorage::Strided, 1usize);
        stages.pipelined(walk, |slot, cell| {
            let mut acc_r = acc_o.at(cell);
            slot.consume(|a_s, b_s| {
                acc_r.mma(a_s, b_s, Semiring::SUM_PROD);
            });
        });
    }
    for r0 in c.over(&outer).unrolled() {
        for r1 in r0.over(&inner).unrolled() {
            let mut c_w = c.at(&r1);
            c_w.copy_cast_from(&acc.at(&r1));
        }
    }
}

// ---- the tensor-core kernels ------------------------------------------------------

/// `c = a · b` through tensor cores over a K walk: the accumulator fragment opened before the
/// walk, both operands staged per region (laid out as `storage`, `depth` in flight), the copy back
/// to global memory the epilogue.
#[cube(launch)]
fn cmma_matmul_k_walk<E: Numeric, V: Size>(
    a: &TileArg<'_, E, V>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.cmma_accumulator::<E, E>(&a, Monoid::Sum);
    acc.zero();
    let walk = space.over(&level);
    let mut stages = Stages::smem(&walk, &a, &b, storage, depth);
    stages.pipelined(walk, |slot, region| {
        let mut acc_r = acc.at(region);
        slot.consume(|a_s, b_s| {
            acc_r.mma(a_s, b_s, Semiring::SUM_PROD);
        });
    });
    for r0 in c.over(&level).unrolled() {
        let mut c_w = c.at(&r0);
        c_w.copy_cast_from(&acc.at(&r0));
    }
}

/// [`cmma_matmul_k_walk`] with a quantized lhs: each region's stage decodes it (or keeps it stored
/// for the fragment load to decode) exactly as the operand states.
#[cube(launch)]
fn cmma_matmul_k_walk_quant<I: Numeric, E: Numeric, V: Size>(
    a: &QuantTileArg<'_, I, V>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] depth: usize,
    #[define(I)] _idtype: ElemType,
    #[define(E)] _edtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.cmma_accumulator::<E, E>(&a, Monoid::Sum);
    acc.zero();
    let walk = space.over(&level);
    let mut stages = Stages::smem(
        &walk,
        &a,
        &b,
        comptime!(StageStorage::Tiled {
            block: Partitioning::new(
                Space::merge(&[&a.place.space, &b.place.space]),
                std::slice::from_ref(&level).to_vec()
            )
            .leaf()
            .extents(),
            chunks: RowChunks::InOrder,
        }),
        depth,
    );
    stages.pipelined(walk, |slot, region| {
        let mut acc_r = acc.at(region);
        slot.consume(|a_s, b_s| {
            acc_r.mma(a_s, b_s, Semiring::SUM_PROD);
        });
    });
    for r0 in c.over(&level).unrolled() {
        let mut c_w = c.at(&r0);
        c_w.copy_cast_from(&acc.at(&r0));
    }
}

/// [`cmma_matmul_k_walk`] through the manual-mma instruction, whose fragment transports are
/// `io`'s.
#[cube(launch)]
fn mma_matmul_k_walk<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] io: MmaIo,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.mma_accumulator::<E, E>(&a, io, Monoid::Sum);
    acc.zero();
    let walk = space.over(&level);
    let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, 1usize);
    stages.pipelined(walk, |slot, region| {
        let mut acc_r = acc.at(region);
        slot.consume(|a_s, b_s| {
            acc_r.mma(a_s, b_s, Semiring::SUM_PROD);
        });
    });
    for r0 in c.over(&level).unrolled() {
        let mut c_w = c.at(&r0);
        c_w.copy_cast_from(&acc.at(&r0));
    }
}

/// [`mma_matmul_k_walk`] with a quantized lhs kept in its stored form by the stage, the manual
/// fragment load decoding each element as it reads.
#[cube(launch)]
fn mma_matmul_k_walk_quant<I: Numeric, E: Numeric>(
    a: &QuantTileArg<'_, I, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] io: MmaIo,
    #[define(I)] _idtype: ElemType,
    #[define(E)] _edtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.mma_accumulator::<E, E>(&a, io, Monoid::Sum);
    acc.zero();
    let walk = space.over(&level);
    let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, 1usize);
    stages.pipelined(walk, |slot, region| {
        let mut acc_r = acc.at(region);
        slot.consume(|a_s, b_s| {
            acc_r.mma(a_s, b_s, Semiring::SUM_PROD);
        });
    });
    for r0 in c.over(&level).unrolled() {
        let mut c_w = c.at(&r0);
        c_w.copy_cast_from(&acc.at(&r0));
    }
}

/// The multi-plane cmma stage: the outer K walk fills a shared stage cooperatively (`depth` in
/// flight), the inner level hands each plane its own fragment of the stage, resident across
/// every K step.
#[cube(launch)]
fn cmma_matmul_two_levels_planes<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.cmma_accumulator::<E, E>(&a, Monoid::Sum);
    acc.zero();
    let walk = space.over(&outer);
    let mut stages = Stages::smem(
        &walk,
        &a,
        &b,
        comptime!(StageStorage::Tiled {
            block: Partitioning::new(
                Space::merge(&[&a.place.space, &b.place.space]),
                vec![outer.clone(), inner.clone()]
            )
            .leaf()
            .extents(),
            chunks: RowChunks::InOrder,
        }),
        depth,
    );
    stages.pipelined(walk, |slot, region| {
        let acc_o = acc.at(region);
        slot.consume(|a_s, b_s| {
            for region in region.over(&inner) {
                let mut acc_p = acc_o.at(&region);
                acc_p.mma(&a_s.at(&region), &b_s.at(&region), Semiring::SUM_PROD);
            }
        });
    });
    for r0 in c.over(&outer).unrolled() {
        for r1 in r0.over(&inner).unrolled() {
            let mut c_w = c.at(&r1);
            c_w.copy_cast_from(&acc.at(&r1));
        }
    }
}

/// The multi-fragment partition: each plane owns a grid of fragments, resident across the outer
/// K walk; the innermost level selects one per region, reloading the operand fragments per
/// execute out of the stage.
#[cube(launch)]
fn cmma_matmul_three_levels_planes_fragments<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] stage: Level,
    #[comptime] plane: Level,
    #[comptime] fragment: Level,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.cmma_accumulator::<E, E>(&a, Monoid::Sum);
    acc.zero();
    let walk = space.over(&stage);
    let mut stages = Stages::smem(
        &walk,
        &a,
        &b,
        comptime!(StageStorage::Tiled {
            block: Partitioning::new(
                Space::merge(&[&a.place.space, &b.place.space]),
                vec![stage.clone(), plane.clone(), fragment.clone()]
            )
            .leaf()
            .extents(),
            chunks: RowChunks::InOrder,
        }),
        depth,
    );
    stages.pipelined(walk, |slot, region| {
        let acc_o = acc.at(region);
        slot.consume(|a_s, b_s| {
            for region in region.over(&plane) {
                let acc_p = acc_o.at(&region);
                let a_p = a_s.at(&region);
                let b_p = b_s.at(&region);
                for frag in region.over(&fragment).unrolled() {
                    let mut acc_f = acc_p.at(&frag);
                    acc_f.mma(&a_p.at(&frag), &b_p.at(&frag), Semiring::SUM_PROD);
                }
            }
        });
    });
    for r0 in c.over(&stage).unrolled() {
        for r1 in r0.over(&plane).unrolled() {
            for r2 in r1.over(&fragment).unrolled() {
                let mut c_w = c.at(&r2);
                c_w.copy_cast_from(&acc.at(&r2));
            }
        }
    }
}

/// The legacy register budget as a level structure: a staged K walk (`depth` in flight), the plane
/// split, a windowing-only step walk, an N walk loading one B fragment per step beside the A
/// column loaded once above, and an M-only fragment walk; both fragment walks unroll (they select).
#[cube(launch)]
fn cmma_matmul_five_levels<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] stage: Level,
    #[comptime] plane: Level,
    #[comptime] step: Level,
    #[comptime] col: Level,
    #[comptime] row: Level,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.cmma_accumulator::<E, E>(&a, Monoid::Sum);
    acc.zero();
    let walk = space.over(&stage);
    let mut stages = Stages::smem(
        &walk,
        &a,
        &b,
        comptime!(StageStorage::Tiled {
            block: Partitioning::new(
                Space::merge(&[&a.place.space, &b.place.space]),
                vec![
                    stage.clone(),
                    plane.clone(),
                    step.clone(),
                    col.clone(),
                    row.clone()
                ]
            )
            .leaf()
            .extents(),
            chunks: RowChunks::InOrder,
        }),
        depth,
    );
    stages.pipelined(walk, |slot, region| {
        let acc_o = acc.at(region);
        slot.consume(|a_s, b_s| {
            for region in region.over(&plane) {
                let acc_p = acc_o.at(&region);
                let a_p = a_s.at(&region);
                let b_p = b_s.at(&region);
                for step in region.over(&step) {
                    let acc_k = acc_p.at(&step);
                    let a_k = a_p.at(&step);
                    let b_k = b_p.at(&step);
                    for col in step.over(&col).unrolled() {
                        let acc_n = acc_k.at(&col);
                        let a_n = PlanePartition::cmma_fragments(&a_k.at(&col), &acc_n);
                        let b_n = PlanePartition::cmma_fragments(&b_k.at(&col), &acc_n);
                        for row in col.over(&row).unrolled() {
                            let mut acc_m = acc_n.at(&row);
                            acc_m.mma(&a_n.at(&row), &b_n.at(&row), Semiring::SUM_PROD);
                        }
                    }
                }
            }
        });
    });
    for r0 in c.over(&stage).unrolled() {
        for r1 in r0.over(&plane).unrolled() {
            for r2 in r1.over(&step).unrolled() {
                for r3 in r2.over(&col).unrolled() {
                    for r4 in r3.over(&row).unrolled() {
                        let mut c_w = c.at(&r4);
                        c_w.copy_cast_from(&acc.at(&r4));
                    }
                }
            }
        }
    }
}

// ---- quantized operands through the register leaf --------------------------------

/// `c = a · b` with a quantized lhs staged in shared memory per region: the stage holds the
/// operand as it states (packed words decoded at the read, or decoded by the fill), and the
/// software instruction runs under `config` out of it.
#[cube(launch)]
fn matmul_quant_lhs_smem_ring<I: Numeric, E: Numeric>(
    a: &QuantTileArg<'_, I, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] config: RegisterBlock,
    #[define(I)] _a_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();
    let walk = space.over(&level);
    let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, 1usize);
    stages.pipelined(walk, |slot, region| {
        let mut c_r = c.at(region);
        slot.consume(|a_s, b_s| {
            c_r.mma_with(a_s, b_s, config, Semiring::SUM_PROD);
        });
    });
}

/// [`matmul_quant_lhs_smem_ring`] serving the quantized lhs straight from global memory, decoded
/// per read.
#[cube(launch)]
fn matmul_quant_lhs_in_place<I: Numeric, E: Numeric, BV: Size>(
    a: &QuantTileArg<'_, I, Const<1>>,
    b: &TileArg<'_, E, BV>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] config: RegisterBlock,
    #[define(I)] _a_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));
    c.zero();
    for region in space.over(&level) {
        let mut c_r = c.at(&region);
        c_r.mma_with(&a.at(&region), &b.at(&region), config, Semiring::SUM_PROD);
    }
}

/// [`matmul_quant_lhs_smem_ring`]'s mirror: the rhs is the quantized operand.
#[cube(launch)]
fn matmul_quant_rhs_smem_ring<I: Numeric, E: Numeric, V: Size>(
    a: &TileArg<'_, E, Const<1>>,
    b: &QuantTileArg<'_, I, Const<1>>,
    c: &TileArg<'_, E, V>,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
    #[comptime] config: RegisterBlock,
    #[define(I)] _b_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile::<E>(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    for outer in space.over(&outer) {
        let a = a.at(&outer);
        let b = b.at(&outer);
        let mut c = c.at(&outer);
        c.zero();
        let walk = outer.over(&inner);
        let mut stages = Stages::smem(&walk, &a, &b, StageStorage::Strided, 1usize);
        stages.pipelined(walk, |slot, region| {
            let mut c_r = c.at(region);
            slot.consume(|a_s, b_s| {
                c_r.mma_with(a_s, b_s, config, Semiring::SUM_PROD);
            });
        });
    }
}

/// [`matmul_quant_lhs_in_place`]'s mirror: the quantized rhs served straight from global memory.
#[cube(launch)]
fn matmul_quant_rhs_in_place<I: Numeric, E: Numeric, V: Size>(
    a: &TileArg<'_, E, Const<1>>,
    b: &QuantTileArg<'_, I, Const<1>>,
    c: &TileArg<'_, E, V>,
    space: Partitioning,
    #[comptime] outer: Level,
    #[comptime] inner: Level,
    #[comptime] config: RegisterBlock,
    #[define(I)] _b_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile::<E>(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    for outer in space.over(&outer) {
        let mut c = c.at(&outer);
        c.zero();
        for region in outer.over(&inner) {
            let mut c_r = c.at(&region);
            c_r.mma_with(&a.at(&region), &b.at(&region), config, Semiring::SUM_PROD);
        }
    }
}

/// [`promoted_matmul_in_place`] with a quantized lhs: a packed lhs contracting into a register
/// block, decoded per read.
#[cube(launch)]
fn promoted_matmul_quant_lhs_in_place<I: Numeric, E: Numeric, EA: Numeric>(
    a: &QuantTileArg<'_, I, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] config: RegisterBlock,
    #[define(I)] _a_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.block_accumulator::<EA, E, E>(&a, &b, config, Monoid::Sum);
    acc.zero();
    for region in space.over(&level) {
        let mut acc_r = acc.at(&region);
        acc_r.mma(&a.at(&region), &b.at(&region), Semiring::SUM_PROD);
    }
    for r0 in c.over(&level).unrolled() {
        let mut c_w = c.at(&r0);
        c_w.copy_cast_from(&acc.at(&r0));
    }
}

// ---- cmma fragment transit, by hand -------------------------------------------------

/// gmem → smem → cmma accumulator → smem → gmem: pure transit, no arithmetic.
#[cube(launch)]
fn cmma_roundtrip<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    output: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let a = input.tile(comptime!(space.clone()));

    let mut a_smem = Tile::shared(comptime!(space.space().clone()), StageStorage::Strided);
    a_smem.copy_from(&a);
    sync_cube();

    let mut frag = CmmaData::<E>::fragment(
        MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(space.space().clone()),
    );
    frag.copy_from(&a_smem);

    let mut c_smem = Tile::shared(comptime!(space.space().clone()), StageStorage::Strided);
    c_smem.copy_from(&frag);
    sync_cube();

    let mut c = output.tile(comptime!(space.clone()));
    c.copy_from(&c_smem);
}

/// gmem A,B → smem → cmma A/B fragments; accumulator init from (zeroed) `c`, then
/// `cmma::execute` (`acc = A·B`), stored back through smem to gmem.
#[cube(launch)]
fn cmma_matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));

    let mut a_smem_tile = Tile::shared(comptime!(a.place.space.clone()), StageStorage::Strided);
    a_smem_tile.copy_from(&a);

    let mut b_smem_tile = Tile::shared(comptime!(b.place.space.clone()), StageStorage::Strided);
    b_smem_tile.copy_from(&b);

    let mut c_smem_tile = Tile::shared(comptime!(c.place.space.clone()), StageStorage::Strided);
    c_smem_tile.copy_from(&c);
    sync_cube();

    let mut a_frag = CmmaData::<E>::fragment(
        MatrixIdent::A,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(a.place.space.clone()),
    );
    a_frag.copy_from(&a_smem_tile);

    let mut b_frag = CmmaData::<E>::fragment(
        MatrixIdent::B,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(b.place.space.clone()),
    );
    b_frag.copy_from(&b_smem_tile);

    let mut acc = CmmaData::<E>::fragment(
        MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(c.place.space.clone()),
    );
    acc.copy_from(&c_smem_tile);

    acc.mma(&a_frag, &b_frag, Semiring::SUM_PROD);

    c_smem_tile.copy_from(&acc);
    sync_cube();
    c.copy_from(&c_smem_tile);
}

/// [`cmma_matmul`] with the rhs stored `{N, K}` and read through a col-major fragment.
#[cube(launch)]
fn cmma_matmul_transposed_rhs<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));

    let mut a_smem = Tile::shared(comptime!(a.place.space.clone()), StageStorage::Strided);
    a_smem.copy_from(&a);

    let mut b_smem = Tile::shared(comptime!(b.place.space.clone()), StageStorage::Strided);
    b_smem.copy_from(&b);
    sync_cube();

    let mut a_frag = CmmaData::<E>::fragment(
        MatrixIdent::A,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(a.place.space.clone()),
    );
    a_frag.copy_from(&a_smem);

    let mut b_frag = CmmaData::<E>::fragment(
        MatrixIdent::B,
        8usize,
        8usize,
        8usize,
        MatrixLayout::ColMajor,
        comptime!(b.place.space.clone()),
    );
    b_frag.copy_from(&b_smem);

    let mut acc = CmmaData::<E>::fragment(
        MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(c.place.space.clone()),
    );
    acc.zero();

    acc.mma(&a_frag, &b_frag, Semiring::SUM_PROD);

    let mut c_smem = Tile::shared(comptime!(c.place.space.clone()), StageStorage::Strided);
    c_smem.copy_from(&acc);
    sync_cube();
    c.copy_from(&c_smem);
}

/// Quantized `A`: gmem `I` (i8) dequantized into smem by the plain `copy_from`, which recovers
/// the storage element from the scheme on its own; `B`/`C` plain `E`. The cmma path then runs
/// entirely in `E`. Mirrors [`cmma_matmul`] otherwise.
#[cube(launch)]
fn cmma_matmul_quant<I: Numeric, E: Numeric>(
    a: &QuantTileArg<'_, I, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[define(I)] _idtype: ElemType,
    #[define(E)] _edtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(comptime!(space.clone()));

    let mut a_smem = Tile::shared(comptime!(a.place.space.clone()), StageStorage::Strided);
    a_smem.copy_from(&a);

    let mut b_smem = Tile::shared(comptime!(b.place.space.clone()), StageStorage::Strided);
    b_smem.copy_from(&b);

    let mut c_smem = Tile::shared(comptime!(c.place.space.clone()), StageStorage::Strided);
    c_smem.copy_from(&c);
    sync_cube();

    let mut a_frag = CmmaData::<E>::fragment(
        MatrixIdent::A,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(a.place.space.clone()),
    );
    a_frag.copy_from(&a_smem);

    let mut b_frag = CmmaData::<E>::fragment(
        MatrixIdent::B,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(b.place.space.clone()),
    );
    b_frag.copy_from(&b_smem);

    let mut acc = CmmaData::<E>::fragment(
        MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(c.place.space.clone()),
    );
    acc.copy_from(&c_smem);

    acc.mma(&a_frag, &b_frag, Semiring::SUM_PROD);

    c_smem.copy_from(&acc);
    sync_cube();
    c.copy_from(&c_smem);
}

// ---- one level, both operands staged ---------------------------------------------

#[test]
fn matmul_sequential_single_cube() {
    check_matmul(
        8,
        8,
        8,
        Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
            .walk_every(&[M, N, K])
            .cubes(&[]),
        1,
    );
}

#[test]
fn matmul_one_tile_per_cube() {
    check_matmul(
        8,
        8,
        8,
        Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
            .walk_every(&[K])
            .cubes(&[M, N]),
        1,
    );
}

/// Cubes distributed their boxes in a swizzled order whose strips the grid does not divide: a 3x5 grid
/// in strips of 2 and of 4 along either axis, the last strip narrower than the rest. Each box must
/// still go to exactly one cube: one distributed twice doubles its product, one distributed to no cube keeps
/// the poison `c` came in with.
#[test]
fn matmul_ragged_swizzle_distributes_every_box_once() {
    for order in [
        CubeOrder::SwizzleRow(2),
        CubeOrder::SwizzleCol(2),
        CubeOrder::SwizzleRow(4),
        CubeOrder::SwizzleCol(4),
    ] {
        check_matmul(
            12,
            20,
            8,
            Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
                .walk_every(&[K])
                .cubes(&[M, N])
                .ordered(order),
            1,
        );
    }
}

/// The kernel owns the init: `c` comes out as `a·b`, whatever it held going in.
///
/// The whole contraction lands at the leaf here, so a single region writes each cell; the poison
/// the harness filled `c` with must be gone all the same.
#[test]
fn matmul_whole_k_at_the_leaf() {
    check_matmul(
        8,
        8,
        4,
        Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
            .walk_every(&[M, N, K])
            .cubes(&[]),
        1,
    );
}

#[test]
fn matmul_reversed_walk_single_cube() {
    let client = cubecl::test_device().client();
    let (m, n, k, tile_edge) = (8usize, 8usize, 8usize, 4usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .uniform(7, -100.0, 100.0);
    matmul_smem_ring_reversed::launch(
        &client,
        launcher.cube_count(),
        CubeDim::new_single(),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        1,
        f32::elem_type_native(),
    );
    assert_tiled_matmul(&client, c.handle(), m, n, k, tile_edge);
}

#[test]
fn matmul_contiguous_m_across_cubes() {
    check_matmul(
        16,
        8,
        8,
        Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
            .walk_every(&[N, K])
            .walk(&[(M, 2)])
            .cubes(&[M]),
        1,
    );
}

#[test]
fn matmul_interleaved_m_across_cubes() {
    check_matmul(
        16,
        8,
        8,
        Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
            .walk_every(&[N, K])
            .cubes(&[M])
            .across(M, 2)
            .interleaved(M),
        1,
    );
}

/// A count the tile grid does not divide: four tiles across three cubes are runs of two, two
/// and nothing; distributed three each they are a run of three and a run of one; in turns they are
/// two, one and one. Every tile is visited once either way, none twice.
#[test]
fn matmul_m_across_cubes_that_do_not_divide() {
    check_matmul(
        16,
        8,
        8,
        Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
            .walk_every(&[N, K])
            .cubes(&[M])
            .across(M, 3),
        1,
    );
}

#[test]
fn matmul_m_three_each_across_cubes_leaves_a_short_run() {
    check_matmul(
        16,
        8,
        8,
        Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
            .walk_every(&[N, K])
            .walk(&[(M, 3)])
            .cubes(&[M]),
        1,
    );
}

#[test]
fn matmul_m_in_turns_across_cubes_that_do_not_divide() {
    check_matmul(
        16,
        8,
        8,
        Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
            .walk_every(&[N, K])
            .cubes(&[M])
            .across(M, 3)
            .interleaved(M),
        1,
    );
}

#[test]
fn matmul_double_buffered() {
    check_matmul(
        8,
        8,
        8,
        Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
            .walk_every(&[M, N, K])
            .cubes(&[]),
        2,
    );
}

/// `c == a·b` for tiled `arange` operands.
fn assert_tiled_matmul(
    client: &Client,
    handle: TensorHandle,
    m: usize,
    n: usize,
    k: usize,
    tile_edge: usize,
) {
    let output = HostData::from_tensor_handle(client, handle, HostDataType::F32);
    let expected = references::tiled_matmul(m, n, k, tile_edge);
    let (_, expected) = TestInput::builder(
        client.clone(),
        shape![m / tile_edge, n / tile_edge, tile_edge, tile_edge],
    )
    .custom(expected)
    .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Drives [`matmul_smem_ring_scheduled`] for `C = A @ B` over `tiling`, whose last level is the
/// walk it stages, `depth` regions in flight under `schedule`, on a cube of `units` units reading
/// every operand in lines `width` wide.
#[allow(clippy::too_many_arguments)]
fn check_matmul_scheduled(
    m: usize,
    n: usize,
    k: usize,
    tiling: Levels,
    depth: usize,
    schedule: Schedule,
    units: u32,
    width: usize,
) {
    let client = cubecl::test_device().client();
    let levels = tiling.build();
    let tile_edge = leaf_edge(&levels, M);
    let space = Space::new(&[(M, m), (N, n), (K, k)]);
    let partitioning = Partitioning::new(space.clone(), levels);
    let grid = Grid::Stated {
        cube_count: partitioning.cube_count(),
        cube_dim: CubeDim::new_1d(units),
    };
    let launcher = Launcher::new(&client, partitioning, &space, grid);
    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .uniform(7, -100.0, 100.0);
    // The cube's units, stated on each operand as `Launcher::arg` states them: a stage fetched
    // into registers distributes its lines over them at expansion. The buffer is bound scalar: the
    // kernel's line type carries the width, and the metadata stays in elements.
    let bound = |input: &TileInput| {
        let mut spec = input.spec();
        spec.units = units as usize;
        TileArgLaunch::new(input.tensor_arg(1), spec)
    };
    matmul_smem_ring_scheduled::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        width,
        bound(&a),
        bound(&b),
        bound(&c),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        depth,
        schedule,
        f32::elem_type_native(),
    );
    assert_tiled_matmul(&client, c.handle(), m, n, k, tile_edge);
}

/// The register-staged schedule against the one it splits, over one slot and two (all it takes),
/// over a `K` walk of four regions a cube and of one: the last region prefetches nothing, and a
/// walk of one region is its prologue alone. Read in scalars and in lines four wide, on one unit,
/// on a few units of one plane, and on a cube of 70 units over stages of 256 elements, whose lines
/// they do not divide: every unit fills its share and the last one contracts, so a missing or
/// misplaced barrier lets it read what units of other planes have not written, or overwrite what
/// it has not read. A case whose cube the device cannot hold is not run: a CPU runtime's cube
/// holds as many units as the host has cores, and a small CI runner has four.
#[test]
fn a_register_staged_ring_matches_a_slot_ahead_ring() {
    // A leaf `edge` wide on every axis, walked along `K`.
    let tiling = |edge: usize| {
        Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
            .walk_every(&[K])
            .cubes(&[M, N])
    };
    // (units, width, problem edge, leaf edge, the `K`s).
    let cases = [
        (1u32, 1, 8, 4, [16, 4]),
        (3, 1, 8, 4, [16, 4]),
        (1, 4, 8, 4, [16, 4]),
        (70, 1, 32, 16, [64, 16]),
        (70, 4, 32, 16, [64, 16]),
    ];
    let max_units = cubecl::test_device()
        .client()
        .properties()
        .hardware
        .max_units_per_cube;
    for (units, width, edge, leaf, ks) in cases {
        if units > max_units {
            continue;
        }
        for k in ks {
            for depth in [1, 2] {
                for schedule in [Schedule::AheadInSlots, Schedule::ThroughRegisters] {
                    check_matmul_scheduled(
                        edge,
                        edge,
                        k,
                        tiling(leaf),
                        depth,
                        schedule,
                        units,
                        width,
                    );
                }
            }
        }
    }
}

/// Drives [`matmul_smem_ring`] for `C = A @ B`: the cube level over the walk it stages, through
/// a run of boxes where `tiling` states one, both inputs staged, `depth` regions in flight.
fn check_matmul(m: usize, n: usize, k: usize, tiling: Levels, depth: usize) {
    let client = cubecl::test_device().client();
    let levels = tiling.build();
    let tile_edge = leaf_edge(&levels, M);
    let launcher = implied(
        &client,
        Partitioning::new(Space::new(&[(M, m), (N, n), (K, k)]), levels),
        Form::Static,
    );
    let (runs, steps) = runs_and_steps(&launcher);

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    // Poisoned, not zeroed: the kernel owns the init, so anything `c` held must be gone from the
    // result.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .uniform(7, -100.0, 100.0);

    matmul_smem_ring::launch(
        &client,
        launcher.cube_count(),
        CubeDim::new_single(),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        runs,
        steps,
        depth,
        f32::elem_type_native(),
    );
    assert_tiled_matmul(&client, c.handle(), m, n, k, tile_edge);
}

/// `mma` never takes the init from the caller: the accumulating kernel folds onto what `c`
/// holds, where the plain one would overwrite.
#[test]
fn mma_folds_onto_what_c_holds() {
    let client = cubecl::test_device().client();
    let (m, n, k, tile_edge) = (8usize, 8usize, 4usize, 4usize);
    // The whole contraction lands at the leaf, where `c = a·b` would overwrite.
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, tile_edge), (N, tile_edge), (K, k)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();

    matmul_smem_ring_accumulate::launch(
        &client,
        launcher.cube_count(),
        CubeDim::new_single(),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        1,
        f32::elem_type_native(),
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // `c` held the same arange the buffer was filled with, cell for cell.
    let expected: Vec<f32> = references::tiled_matmul(m, n, k, tile_edge)
        .into_iter()
        .enumerate()
        .map(|(i, product)| product + i as f32)
        .collect();
    let (_, expected) = TestInput::builder(
        client,
        shape![m / tile_edge, n / tile_edge, tile_edge, tile_edge],
    )
    .custom(expected)
    .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

// ---- batches and broadcasts ----------------------------------------------------------

#[test]
fn matmul_batched_walked() {
    check_matmul_batched(3, 8, 8, 8, 4, 1);
}

#[test]
fn matmul_batched_in_sub_tile() {
    check_matmul_batched(4, 8, 8, 8, 4, 4);
}

#[test]
fn matmul_batched_split() {
    check_matmul_batched(4, 8, 8, 8, 4, 2);
}

fn check_matmul_batched(
    b: usize,
    m: usize,
    n: usize,
    k: usize,
    tile_edge: usize,
    batch_edge: usize,
) {
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(B, b), (M, m), (N, n), (K, k)]),
            Levels::leaf(&[
                (B, batch_edge),
                (M, tile_edge),
                (N, tile_edge),
                (K, tile_edge),
            ])
            .walk_every(&[B, M, N, K])
            .build(),
        ),
        Form::Static,
    );
    let a = TileInput::builder(&client, launcher.space().subspace(&[B, M, K]))
        .tile(&[batch_edge, tile_edge, tile_edge])
        .arange();
    let rhs = TileInput::builder(&client, launcher.space().subspace(&[B, K, N]))
        .tile(&[batch_edge, tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[B, M, N]))
        .tile(&[batch_edge, tile_edge, tile_edge])
        .zeros();

    matmul_smem_ring::launch(
        &client,
        launcher.cube_count(),
        CubeDim::new_single(),
        1,
        a.arg(),
        rhs.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        one_cube(),
        None,
        launcher.partitioning().level(0),
        1,
        f32::elem_type_native(),
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected = references::batched_tiled_matmul(b, m, n, k, tile_edge, batch_edge);
    let (grid_m, grid_n) = (m / tile_edge, n / tile_edge);
    let (_, expected) = TestInput::builder(
        client,
        shape![
            b / batch_edge,
            grid_m,
            grid_n,
            batch_edge,
            tile_edge,
            tile_edge
        ],
    )
    .custom(expected)
    .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Selective batch broadcast over two batch axes `B0 = b0`, `B1 = b1`: `lhs` carries
/// `B0` (and broadcasts `B1`), `rhs` carries `B1` (and broadcasts `B0`). The merge
/// rebuilds the full `{B0, B1}` output batch so every operand reads the right slice.
#[test]
fn matmul_broadcast_two_batch_axes() {
    check_matmul_broadcast(
        4,
        3,
        4,
        &Levels::leaf(&[(B0, 1), (B1, 1), (M, 4), (N, 4), (K, 4)])
            .walk_every(&[B0, B1, M, N, K])
            .build(),
    );
}

#[test]
fn matmul_broadcast_lhs_only() {
    // rhs broadcasts nothing (b0 = 1 makes B0 degenerate); lhs still broadcasts B1.
    check_matmul_broadcast(
        1,
        5,
        4,
        &Levels::leaf(&[(B0, 1), (B1, 1), (M, 4), (N, 4), (K, 4)])
            .walk_every(&[B0, B1, M, N, K])
            .build(),
    );
}

/// Both batch axes ride cube-Z at once: `B0` and `B1` are `Spatial { Cube(Z) }`, so the launch
/// puts their *product* on Z and the walk decodes one cube's `CUBE_POS_Z` back into `(b0, b1)`.
/// Same broadcast result as the sequential variants; this lets CpuGemm parallelise a batch on Z.
#[test]
fn matmul_broadcast_two_batch_axes_on_z() {
    check_matmul_broadcast(
        4,
        3,
        4,
        &Levels::leaf(&[(B0, 1), (B1, 1), (M, 4), (N, 4), (K, 4)])
            .walk_every(&[M, N, K])
            .cubes(&[B0, B1])
            .build(),
    );
}

/// The two-axis broadcast tiled across *two* levels: L0 walks the batch (`batch_edge = 1`) and
/// stages the whole `4×4` matrix, then L1 tiles that matrix into `2×2` final tiles. The broadcast
/// (omitted) batch axes must stay correct through both `divide`s; the result is the same matmul.
#[test]
fn matmul_broadcast_multilevel() {
    check_matmul_broadcast(
        4,
        3,
        4,
        &Levels::leaf(&[(B0, 1), (B1, 1), (M, 2), (N, 2), (K, 2)])
            .walk(&[(B0, 1), (B1, 1), (M, 2), (N, 2), (K, 2)])
            .walk_every(&[B0, B1, M, N, K])
            .build(),
    );
}

/// `C = A @ B` where the batch is two independent axes `B0`, `B1` and each operand carries only
/// one: `lhs ∈ {B0, M, K}`, `rhs ∈ {B1, K, N}`, `out ∈ {B0, B1, M, N}`. Each operand omits the
/// batch axis it broadcasts, and the kernel's `Space::merge` fills the omitted axis back wholesale.
///
/// Single tile per matrix (`t³`) with `batch_edge = 1`, so each output batch element is its own
/// walk point. Every level stages, whatever the caller stacked.
fn check_matmul_broadcast(b0: usize, b1: usize, t: usize, levels: &[Level]) {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(B0, b0), (B1, b1), (M, t), (N, t), (K, t)]),
            levels.to_vec(),
        ),
        Form::Static,
    );
    let out = launcher.space().subspace(&[B0, B1, M, N]);
    let lhs = TileInput::builder(&client, launcher.space().subspace(&[B0, M, K]))
        .tile(&[1, t, t])
        .arange();
    let rhs = TileInput::builder(&client, launcher.space().subspace(&[B1, K, N]))
        .tile(&[1, t, t])
        .arange();
    let acc = TileInput::builder(&client, out.clone())
        .tile(&[1, 1, t, t])
        .zeros();

    let cube_count = launcher.cube_count();
    let cube_dim = CubeDim::new_single();
    match levels.len() {
        1 => matmul_smem_ring::launch(
            &client,
            cube_count,
            cube_dim,
            1,
            lhs.arg(),
            rhs.arg(),
            acc.arg(),
            launcher.partitioning_arg(),
            one_cube(),
            None,
            launcher.partitioning().level(0),
            1,
            dtype,
        ),
        2 => matmul_two_levels_smem_then_smem::launch(
            &client,
            cube_count,
            cube_dim,
            lhs.arg(),
            rhs.arg(),
            acc.arg(),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            launcher.partitioning().level(1),
            StageStorage::Strided,
            1,
            1,
            dtype,
        ),
        depth => panic!("check_matmul_broadcast: no kernel walks {depth} levels"),
    }

    let output = HostData::from_tensor_handle(&client, acc.handle(), HostDataType::F32);
    let expected = references::broadcast_matmul(b0, b1, t);
    let (_, expected) = TestInput::builder(client, shape![b0, b1, 1, 1, 1, 1, t, t])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

// ---- one level, every operand where it lies ---------------------------------------

#[test]
fn matmul_cpu_sequential() {
    check_matmul_cpu(
        8,
        8,
        8,
        Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
            .walk_every(&[M, N, K])
            .cubes(&[])
            .build(),
    );
}

#[test]
fn matmul_cpu_big_k() {
    check_matmul_cpu(
        8,
        8,
        16,
        Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
            .walk_every(&[M, N, K])
            .cubes(&[])
            .build(),
    );
}

#[test]
fn matmul_cpu_cores_split_m() {
    check_matmul_cpu(
        16,
        8,
        8,
        Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
            .walk_every(&[N, K])
            .walk(&[(M, 2)])
            .cubes(&[M])
            .build(),
    );
}

/// Each plane walks a run of two tiles: the planes are as many as `m` holds such runs.
#[test]
fn matmul_cpu_cores_split_m_planes() {
    let (m, tiles_each) = (16usize, 2usize);
    check_matmul_cpu(
        m,
        8,
        8,
        Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
            .walk_every(&[N, K])
            .walk(&[(M, tiles_each)])
            .planes(&[(M, m / (4 * tiles_each))])
            .build(),
    );
}

// Short runs across a cube's planes (four tiles over three planes, three each, in turns) cannot be
// stated any more: a plane level says how many planes take one tile each, and a run of tiles is a
// walk below it, taken whole by every plane; only a cube level distributes runs (`Levels::across`).

/// The register leaf reads both operands where they lie: nothing is materialized and the walk is
/// the plain loop. `levels` is the worker level over the walk, with a run of boxes between them
/// where the tiling states one.
fn check_matmul_cpu(m: usize, n: usize, k: usize, levels: Vec<Level>) {
    let client = cubecl::test_device().client();
    let tile_edge = leaf_edge(&levels, M);
    let launcher = implied(
        &client,
        Partitioning::new(Space::new(&[(M, m), (N, n), (K, k)]), levels),
        Form::Static,
    );
    let (runs, inner) = runs_and_steps(&launcher);

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    // Poisoned, not zeroed: the kernel owns `out = A·B` whatever the buffer held.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .uniform(4242, 10., 100.);

    matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        runs,
        inner,
        REGISTER_BLOCK,
        Semiring::SUM_PROD,
        f32::elem_type_native(),
    );
    assert_tiled_matmul(&client, c.handle(), m, n, k, tile_edge);
}

/// The "global matmul" shape: M and N stay comptime (`Static`) and only K is `Dynamic`, resolved
/// from the tensor at runtime while M/N fold and unroll: the mixed `merged_space`/`Extents` path
/// that `all_dynamic` skips. Allocation uses the concrete nest, the kernel the K-dynamic one.
#[test]
fn matmul_cpu_dynamic_k() {
    let client = cubecl::test_device().client();
    let (m, n, k, edge) = (8usize, 8usize, 16usize, 4usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .tile(&[edge, edge])
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .tile(&[edge, edge])
        .arange();
    // Poisoned, not zeroed: the kernel owns `out = A·B` whatever the buffer held.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .tile(&[edge, edge])
        .uniform(4242, 10., 100.);

    matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        uncut(
            &client,
            &launcher.space().clone().with_dynamic(&[K]),
            launcher.space(),
        )
        .partitioning_arg(),
        one_cube(),
        None,
        launcher.partitioning().level(0),
        REGISTER_BLOCK,
        Semiring::SUM_PROD,
        f32::elem_type_native(),
    );
    assert_tiled_matmul(&client, c.handle(), m, n, k, edge);
}

/// N spread across a plane's units (`ComputeScope::Unit`): each unit owns a disjoint column of
/// the register-leaf output and contracts all of K in registers: the gemv-perpendicular mapping.
/// `plane_size == 1` on CPU is one unit doing all of N (still correct); the win is on GPU's units.
///
/// A bare `units()` declares the split without the unit count; [`Space::resolve_units`] (the
/// launch's stamping pass) fills it from the hardware `plane_size`, so the Unit axis rides the
/// warp's units on the cube's X dim.
#[test]
fn register_matmul_unit_spread_n() {
    let client = cubecl::test_device().client();
    let plane_size = client.properties().hardware.plane_size_max as usize;

    let (m, k, nr) = (4usize, 8usize, 2usize);
    let n = plane_size * nr;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(N, nr)]).units(&[(N, plane_size)]).build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .zeros();

    matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        one_cube(),
        None,
        launcher.partitioning().level(0),
        REGISTER_BLOCK,
        Semiring::SUM_PROD,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

// ---- padded stages ----------------------------------------------------------------

/// A scalar `K×N` source whose `N = 3` rows cannot be vectorized globally is padded into
/// four-wide shared-memory lines. The register gather scatters those units back into the scalar
/// `M×N` sink; the fourth unit is padding and must not consume the next row's first live value.
#[test]
fn matmul_padded_rhs_stage_into_scalar_sink() {
    check_padded_rhs_stage((2, 3, 2), vec![3.0, 4.0, 5.0, 9.0, 14.0, 19.0]);
}

/// The single-row shape, where a block column overhanging `N` has nowhere legal to land: with no
/// row after it, `block::commit`'s masked units are all that keeps the write inside the output.
#[test]
fn matmul_padded_rhs_stage_single_row_sink() {
    check_padded_rhs_stage((1, 3, 2), vec![3.0, 4.0, 5.0]);
}

/// A scalar `K×N` source with `N = 5` spanning two 4-wide shared-memory lines (total 8 units, 3
/// padding). Exercises non-multiple tail across multi-line stages.
#[test]
fn matmul_padded_rhs_stage_multi_line() {
    check_padded_rhs_stage(
        (2, 5, 2),
        vec![5.0, 6.0, 7.0, 8.0, 9.0, 15.0, 20.0, 25.0, 30.0, 35.0],
    );
}

fn check_padded_rhs_stage((m, n, k): (usize, usize, usize), expected: Vec<f32>) {
    let client = cubecl::test_device().client();
    let launch = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, m), (N, n), (K, k)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    let a = TileInput::builder(&client, launch.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launch.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launch.space().subspace(&[M, N]))
        .untiled()
        .zeros();
    let launcher = implied(
        &client,
        Partitioning::new(
            launch.space().clone(),
            launch.partitioning().levels().to_vec(),
        ),
        Form::Dynamic,
    );
    let a_op = launcher.arg(a.handle().binding()).axes(&[M, K]).build();
    let b_op = launcher.arg(b.handle().binding()).axes(&[K, N]).build();
    let c_op = launcher.arg(c.handle().binding()).axes(&[M, N]).build();

    matmul_padded_rhs_stage::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a_op.arg(),
        b_op.arg(),
        c_op.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        4,
        f32::elem_type_native(),
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// A scalar `M×K` source with `K = 3` passes an outer level read where it lies before the inner
/// one pads it into one four-wide line per row. Unlike the padded-rhs cases above, a scalar rhs
/// and sink keep it on the direct 2-D path; its partial last lhs line must add exactly 3 K values.
#[test]
fn matmul_padded_lhs_stage_direct_tail() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (2usize, 2usize, 3usize);
    let launch = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, m), (N, n), (K, k)])
                .walk(&[(M, 1), (N, 1), (K, 1)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    let a = TileInput::builder(&client, launch.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launch.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launch.space().subspace(&[M, N]))
        .untiled()
        .zeros();
    let launcher = implied(
        &client,
        Partitioning::new(
            launch.space().clone(),
            launch.partitioning().levels().to_vec(),
        ),
        Form::Dynamic,
    );
    let a_op = launcher.arg(a.handle().binding()).axes(&[M, K]).build();
    let b_op = launcher.arg(b.handle().binding()).axes(&[K, N]).build();
    let c_op = launcher.arg(c.handle().binding()).axes(&[M, N]).build();

    matmul_padded_lhs_stage_two_levels::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a_op.arg(),
        b_op.arg(),
        c_op.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        4,
        f32::elem_type_native(),
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(vec![10.0, 13.0, 28.0, 40.0])
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

// ---- two levels ------------------------------------------------------------------

/// Two levels stacked: the outer stages `4×4×4` blocks, the inner walks `2×2×2` final tiles
/// of the stage last to first, where they lie.
#[test]
fn matmul_multilevel_staged_then_direct() {
    let client = cubecl::test_device().client();
    let (m, n, k, final_edge) = (8usize, 8usize, 8usize, 2usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, 2), (N, 2), (K, 2)])
                .walk(&[(M, 2), (N, 2), (K, 2)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .tile(&[final_edge, final_edge])
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .tile(&[final_edge, final_edge])
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .tile(&[final_edge, final_edge])
        .zeros();
    matmul_two_levels_smem_then_in_place_reversed::launch(
        &client,
        launcher.cube_count(),
        CubeDim::new_single(),
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        StageStorage::Strided,
        1,
        f32::elem_type_native(),
    );
    assert_tiled_matmul(&client, c.handle(), m, n, k, final_edge);
}

#[test]
fn matmul_multilevel_staged_then_staged() {
    check_matmul_multilevel(8, 8, 8, StageLayout::Strided, Inner::Staged(1), 1);
}

/// Double buffering at the higher level.
#[test]
fn matmul_multilevel_double_then_direct() {
    check_matmul_multilevel(8, 8, 8, StageLayout::Strided, Inner::Direct, 2);
}

/// Double buffering at the lower level.
#[test]
fn matmul_multilevel_staged_then_double() {
    check_matmul_multilevel(8, 8, 8, StageLayout::Strided, Inner::Staged(2), 1);
}

/// A storage-tiled stage on a register leaf: the stage layout knob off its default,
/// on any backend (each 4×4 stage cut into contiguous 2×2 blocks).
#[test]
fn matmul_multilevel_tiled_stage() {
    check_matmul_multilevel(8, 8, 8, StageLayout::Tiled, Inner::Direct, 1);
}

/// What the inner of two levels does with the outer stage: read its final tiles where they lie,
/// or restage them through stages this deep.
#[derive(Clone, Copy)]
enum Inner {
    Direct,
    Staged(usize),
}

/// Drives the two-level kernels over `[4×4×4, 2×2×2]`: the outer level stages both operands laid
/// out as `storage`, `depth_outer` regions in flight; `inner` says what the second level does.
/// How a test's stages lay their buffers out, resolved against the space the test builds.
#[derive(Clone, Copy)]
enum StageLayout {
    Tiled,
    Strided,
}

impl StageLayout {
    fn storage(self, launcher: &Launcher) -> StageStorage {
        match self {
            StageLayout::Tiled => StageStorage::Tiled {
                block: launcher.partitioning().leaf().extents(),
                chunks: RowChunks::InOrder,
            },
            StageLayout::Strided => StageStorage::Strided,
        }
    }
}

fn check_matmul_multilevel(
    m: usize,
    n: usize,
    k: usize,
    layout: StageLayout,
    inner: Inner,
    depth_outer: usize,
) {
    let client = cubecl::test_device().client();
    let final_edge = 2usize;
    let dtype = f32::elem_type_native();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, 2), (N, 2), (K, 2)])
                .walk(&[(M, 2), (N, 2), (K, 2)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    let storage = layout.storage(&launcher);

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .tile(&[final_edge, final_edge])
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .tile(&[final_edge, final_edge])
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .tile(&[final_edge, final_edge])
        .zeros();

    match inner {
        Inner::Direct => matmul_two_levels_smem_then_in_place::launch(
            &client,
            launcher.cube_count(),
            CubeDim::new_single(),
            a.arg(),
            b.arg(),
            c.arg(),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            launcher.partitioning().level(1),
            storage,
            depth_outer,
            dtype,
        ),
        Inner::Staged(depth_inner) => matmul_two_levels_smem_then_smem::launch(
            &client,
            launcher.cube_count(),
            CubeDim::new_single(),
            a.arg(),
            b.arg(),
            c.arg(),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            launcher.partitioning().level(1),
            storage,
            depth_outer,
            depth_inner,
            dtype,
        ),
    }
    assert_tiled_matmul(&client, c.handle(), m, n, k, final_edge);
}

/// A staged level whose walk leaves the lhs unchanged (an N-only walk at L1): the
/// invariant operand fills its slot once, above the loop.
#[test]
fn matmul_staged_invariant_lhs() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (8usize, 8usize, 8usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, 4), (N, 2), (K, 4)])
                .walk(&[(M, 1), (N, 2), (K, 1)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .zeros();

    matmul_two_levels_smem_then_smem::launch(
        &client,
        launcher.cube_count(),
        CubeDim::new_single(),
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        StageStorage::Strided,
        1,
        1,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// A level whose edges equal the extents handed to it partitions nothing, and the build keeps it
/// all the same: the kernel walks exactly the levels stated, and a one-region level folds to no
/// loop at all. The two-level kernel over it computes what the one-level one does.
#[test]
fn matmul_a_level_that_cuts_nothing_is_kept() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (8usize, 8usize, 8usize);
    let plain = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    // The second level's edges are the first's: every axis's count is 1.
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
                .walk(&[(M, 1), (N, 1), (K, 1)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    assert_eq!(plain.partitioning().levels().len(), 1);
    assert_eq!(launcher.partitioning().levels().len(), 2);
    assert_ne!(
        launcher.partitioning().levels(),
        plain.partitioning().levels()
    );
    assert_eq!(launcher.cube_dim(), plain.cube_dim());

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .zeros();

    matmul_two_levels_smem_then_in_place::launch(
        &client,
        launcher.cube_count(),
        CubeDim::new_single(),
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        StageStorage::Strided,
        1,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

// A contraction cut at cube scope leaves each cube holding a slice of every output cell, and
// nothing combines them: a register accumulator drains by storing, so the last cube to arrive
// erases the others, and one accumulating in place reads, folds and writes back: a lost update.
//
// Both are refused (`SplitShare::validate`), and the refusal is unit-tested where it can be
// observed, `space::base`. Not here, for the reason `blocked.rs` gives: this one fires inside the
// kernel, on a worker thread, where `#[should_panic]` never sees it and the launch returns zeros.

// ---- vectorized operands through the stages -------------------------------------------

/// Vectorized operands (2-wide lines) through the in-place path: gmem-only line-unit
/// addressing. Regression for the line-vs-scalar unit bug (worked on cubecl-cpu only).
#[test]
fn matmul_direct_vectorized() {
    let client = cubecl::test_device().client();
    let (m, n, k, edge) = (8usize, 8usize, 8usize, 4usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .zeros();
    matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        CubeDim::new_single(),
        2,
        2,
        2,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        one_cube(),
        None,
        launcher.partitioning().level(0),
        REGISTER_BLOCK,
        Semiring::SUM_PROD,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// The same walk with the operands staged instead: the cooperative fill moves lines through smem.
/// Regression for the line-vs-scalar unit bug.
#[test]
fn matmul_staged_vectorized() {
    check_matmul_vectorized((8, 8, 8), Staged::Both, 1);
}

/// The same operands through a depth-2 stages: each region's fill overlaps the previous region's
/// compute. Depth is the only difference from the staged case above.
#[test]
fn matmul_double_buffered_vectorized() {
    check_matmul_vectorized((8, 8, 8), Staged::Both, 2);
}

/// Depth 3: two fills in flight over one compute. Regression for stages whose drain leaves more
/// than one slot outstanding.
#[test]
fn matmul_triple_buffered_vectorized() {
    check_matmul_vectorized((8, 8, 8), Staged::Both, 3);
}

/// A depth deeper than the walk has regions: the prologue runs out of regions to prime and every
/// consume drains. Regression for the stages' `region < total` guards. Nine slots over eight
/// regions: two operands per slot, and Metal caps a kernel's threadgroup arguments at 31.
#[test]
fn matmul_buffered_deeper_than_the_walk() {
    check_matmul_vectorized((8, 8, 8), Staged::Both, 9);
}

/// A depth-2 stages whose walk cuts only `M`: `rhs` spans `K`/`N` alone, so the walk never moves its
/// window. It is filled once above the loop and its buffer serves both slots, `Refill::Shared`:
/// the one sound way for two slots to share a buffer, and why a stage count is derived, not stated.
#[test]
fn matmul_double_buffered_with_a_fixed_operand() {
    check_matmul_vectorized((8, 4, 4), Staged::Both, 2);
}

/// The same fixed operand three slots deep, so two slots reuse the first slot's buffer.
#[test]
fn matmul_triple_buffered_with_a_fixed_operand() {
    check_matmul_vectorized((8, 4, 4), Staged::Both, 3);
}

/// One operand staged beside one read where it lies, at depth 2: the slot rendezvouses for the
/// staged one alone while the other is read where it lies, in every slot of the stages.
#[test]
fn matmul_double_buffered_mixed_residence_vectorized() {
    check_matmul_vectorized((8, 8, 8), Staged::LhsOnly, 2);
}

/// Double buffering with only the lhs staged, on tiled buffers: `a` takes a shared stage while
/// `b` is read straight from global memory, in one slot, on a level that prefetches.
#[test]
fn matmul_double_buffered_with_only_the_lhs_staged() {
    let client = cubecl::test_device().client();
    let (m, n, k, tile_edge) = (8usize, 8usize, 8usize, 4usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, tile_edge), (N, tile_edge), (K, tile_edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    // Poisoned, not zeroed: the kernel owns the init.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .uniform(7, -100.0, 100.0);

    matmul_lhs_smem_ring::launch(
        &client,
        launcher.cube_count(),
        CubeDim::new_single(),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        2,
        f32::elem_type_native(),
    );
    assert_tiled_matmul(&client, c.handle(), m, n, k, tile_edge);
}

/// Which operands are staged.
#[derive(Clone, Copy)]
enum Staged {
    Both,
    LhsOnly,
}

fn check_matmul_vectorized((m, n, k): (usize, usize, usize), staged: Staged, depth: usize) {
    let client = cubecl::test_device().client();
    let (edge, v) = (4usize, 2usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let dtype = f32::elem_type_native();
    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .zeros();

    match staged {
        Staged::Both => matmul_smem_ring::launch(
            &client,
            launcher.cube_count(),
            CubeDim::new_single(),
            v,
            a.arg(),
            b.arg(),
            c.arg(),
            launcher.partitioning_arg(),
            one_cube(),
            None,
            launcher.partitioning().level(0),
            depth,
            dtype,
        ),
        Staged::LhsOnly => matmul_lhs_smem_ring::launch(
            &client,
            launcher.cube_count(),
            CubeDim::new_single(),
            v,
            a.arg(),
            b.arg(),
            c.arg(),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            depth,
            dtype,
        ),
    }
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

// ---- the promoted register block ---------------------------------------------------

/// The register leaf contracts through a promoted block rather than through the output, so a
/// deep `K` keeps its partials in the accumulate element instead of round-tripping them
/// through the sink's on every visit.
#[test]
fn register_matmul_promoted_accumulator() {
    let client = cubecl::test_device().client();
    // One block per instance (a 1x1 partition at the leaf), K walked in four steps: every
    // step returns to the same promoted accumulator, which is the round trip this removes.
    let (m, n, k, edge) = (4usize, 4usize, 16usize, 4usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    // Poisoned: the kernel owns `out = A·B` whatever the buffer held.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let dtype = f32::elem_type_native();
    promoted_matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        REGISTER_BLOCK,
        Semiring::SUM_PROD,
        dtype,
        dtype,
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// Min-plus in place: the memory nest seeds `+∞`, forms `a + b` and folds under `min`, all of it
/// off the semiring the contraction was handed. Random operands, so the winning `p` differs cell
/// by cell and a leaf that kept the ordinary `fma` cannot land on these numbers.
#[test]
fn tropical_matmul_in_place() {
    let client = cubecl::test_device().client();
    let (m, n, k, edge) = (4usize, 4usize, 8usize, 4usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .uniform(7, 1., 9.);
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .uniform(13, 1., 9.);
    // Poisoned: the kernel owns the init under this algebra too, and its identity is not zero.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        one_cube(),
        None,
        launcher.partitioning().level(0),
        REGISTER_BLOCK,
        Semiring::MIN_SUM,
        f32::elem_type_native(),
    );

    let lhs = HostData::from_tensor_handle(&client, a.handle(), HostDataType::F32);
    let rhs = HostData::from_tensor_handle(&client, b.handle(), HostDataType::F32);
    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| lhs.get_f32(&[i, p]) + rhs.get_f32(&[p, j]))
                .fold(f32::INFINITY, f32::min)
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Max-plus through a promoted block: the register accumulator is built under `Max`, so it starts
/// at the lowest value, steps with `+`, and drains its units under the same fold.
#[test]
fn tropical_matmul_promoted() {
    let client = cubecl::test_device().client();
    let (m, n, k, edge) = (4usize, 4usize, 8usize, 4usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .uniform(11, 1., 9.);
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .uniform(17, 1., 9.);
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let dtype = f32::elem_type_native();
    promoted_matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        REGISTER_BLOCK,
        Semiring::MAX_SUM,
        dtype,
        dtype,
    );

    let lhs = HostData::from_tensor_handle(&client, a.handle(), HostDataType::F32);
    let rhs = HostData::from_tensor_handle(&client, b.handle(), HostDataType::F32);
    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| lhs.get_f32(&[i, p]) + rhs.get_f32(&[p, j]))
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The promoted register accumulator under the two-level cube/plane nest a real gemm composes,
/// with **vectorized** operands (rhs and output in 2-wide lines).
///
/// This once failed to compile on the CPU backend, when the block was allocated scalar and
/// re-viewed as lines; the block is now allocated at its vector element (`Array<Vector<T, RA>>`),
/// so the store is a real vector write and the numbers are right on every runtime.
#[test]
fn register_matmul_promoted_cube_plane() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (4usize, 4usize, 16usize);
    let (leaf_m, leaf_n, leaf_k) = (2usize, 2usize, 4usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, leaf_m), (N, leaf_n), (K, leaf_k)])
                .walk_every(&[K])
                .planes(&[(M, m / leaf_m), (N, n / leaf_n)])
                .cubes(&[M, N])
                .build(),
        ),
        Form::Static,
    );

    let dtype = f32::elem_type_native();
    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    promoted_matmul_two_levels_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        // Rhs and output vectorized along N, as a real launch does: the tensor args stay
        // scalar-unit and the kernel's `Vector<E, V>` element carries the width.
        2,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        REGISTER_BLOCK,
        dtype,
        dtype,
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// One body, whatever a plane contracts through: the instruction is an argument, and both
/// [`Tile::accumulator`] and [`PlanePartition::operand`] take it.
///
/// The kernel a derivation writes when it elects a form from what the device offers. It matches
/// on nothing: the register form reads its operands out of the tiles it was handed and the
/// matrix forms load fragments, and that difference is the engine's.
#[cube(launch)]
fn matmul_on_a_stated_instruction<E: Numeric, EA: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    space: Partitioning,
    #[comptime] instruction: Instruction,
    #[define(E)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    for cube in space {
        for plane in cube {
            let c_p = c.at(&plane);
            let a_p = a.at(&plane);
            let b_p = b.at(&plane);
            let mut acc = c_p.accumulator::<EA, E, E>(&a_p, &b_p, instruction, Monoid::Sum);
            acc.zero();
            for step in plane {
                let mut acc_s = acc.at(&step);
                let a_f = PlanePartition::<E>::operand(&a_p.at(&step), &acc_s, instruction);
                let b_f = PlanePartition::<E>::operand(&b_p.at(&step), &acc_s, instruction);
                acc_s.mma(&a_f, &b_f, Semiring::SUM_PROD);
            }
            for r0 in c_p.walk().unrolled() {
                let mut c_p_w = c_p.at(&r0);
                c_p_w.copy_cast_from(&acc.at(&r0));
            }
        }
    }
}

/// The register form of [`matmul_on_a_stated_instruction`], which every device runs.
///
/// What it proves is that the pair is form-blind on the side that asks nothing of the hardware:
/// `operand` hands a register block the tile it was given, so a kernel written for a fragment
/// form runs unchanged without one; the cmma form is [`instruction_stated_once_runs_on_cmma`].
#[test]
fn instruction_stated_once_runs_in_registers() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (4usize, 4usize, 16usize);
    let (leaf_m, leaf_n, leaf_k) = (2usize, 2usize, 4usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, leaf_m), (N, leaf_n), (K, leaf_k)])
                .walk_every(&[K])
                .planes(&[(M, m / leaf_m), (N, n / leaf_n)])
                .cubes(&[M, N])
                .build(),
        ),
        Form::Static,
    );

    let dtype = f32::elem_type_native();
    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    matmul_on_a_stated_instruction::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        Instruction::Registers {
            config: REGISTER_BLOCK,
        },
        dtype,
        dtype,
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// A buffered level that *cuts* a promoted (fragment) accumulator: each region selects its own
/// block, so the stages' walk has to unroll and hand every region comptime coordinates.
///
/// The regression this guards is silent in both directions: `#[unroll(flag)]` rolls the loop
/// without complaint unless the macro sees `flag` as a comptime binding, and the lap arithmetic
/// must fold or the coordinates come out runtime even unrolled. Either slip panics in `Tile::at`.
///
/// The other unrolled shape, a fragment stage, needs a fragment leaf and so only runs on
/// tensor-core hardware ([`cmma_matmul_staged_n_walk_partition`]); this one runs everywhere.
#[test]
fn matmul_buffered_walk_cutting_a_fragment_accumulator_unrolls() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (4usize, 4usize, 8usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, 2), (N, 2), (K, 2)])
                .walk(&[(M, 2), (N, 2), (K, 2)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel zeroes the promoted accumulator.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    block_matmul_two_levels_smem_below::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

// ---- lined lhs: the (line, component) K walk --------------------------------------

/// A single-level nest whose leaf takes the whole problem, the shape the lined-lhs and folded
/// tests drive.
fn lined_lhs_space(m: usize, n: usize, k: usize) -> Launcher {
    implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, m), (N, n), (K, k)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    )
}

/// The memory-backed leaf with the lhs lined 2-wide along `K`: two units per K-line, each
/// reaching its element by a comptime `extract` rather than a dynamic one.
#[test]
fn register_matmul_lined_lhs() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (4usize, 4usize, 8usize);
    let launcher = lined_lhs_space(m, n, k);

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel owns `out = A·B` whatever the buffer held.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        2,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        one_cube(),
        None,
        launcher.partitioning().level(0),
        REGISTER_BLOCK,
        Semiring::SUM_PROD,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// [`register_matmul_lined_lhs`] through the promoted block: same walk, same units, but the
/// accumulator never round-trips to the output between `K` steps.
#[test]
fn register_matmul_promoted_lined_lhs() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (4usize, 4usize, 8usize);
    let launcher = lined_lhs_space(m, n, k);

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let dtype = f32::elem_type_native();
    promoted_matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        // Lhs 2-wide along K, rhs and output 2-wide along N: both the unit fan-out and the
        // block's own line width are off their scalar case at once.
        2,
        2,
        2,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        REGISTER_BLOCK,
        Semiring::SUM_PROD,
        dtype,
        dtype,
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

// ---- folded step: both operands lined along K --------------------------------

/// `A·Bᵀ` off row-major `arange` operands: `lhs(i, p) = i·k + p`, `rhs(j, p) = j·k + p`.
fn folded_matmul_reference(m: usize, n: usize, k: usize) -> Vec<f32> {
    (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k).map(|p| ((i * k + p) * (j * k + p)) as f32).sum()
        })
        .collect()
}

/// Both operands lined along `K` with a scalar output: a step consumes a whole line, the block's
/// units are `K`-partials of one cell, and one horizontal fold collapses them. The rhs is declared
/// `[N, K]`, which puts its line on the contracted axis.
///
/// `budget` sizes the block: too small for the shape and the rolled body runs, indexing its local
/// arrays at runtime.
fn check_folded_step(launcher: Launcher, (m, n, k): (usize, usize, usize), budget: usize) {
    let client = cubecl::test_device().client();
    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[N, K]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel owns `out = A·Bᵀ` whatever the buffer held.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        4,
        4,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        one_cube(),
        None,
        launcher.partitioning().level(0),
        RegisterBlock::new(budget),
        Semiring::SUM_PROD,
        f32::elem_type_native(),
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(folded_matmul_reference(m, n, k))
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The 2-D nest at a folded step: four contracted values per `fma` instead of one.
#[test]
fn register_matmul_folded_step() {
    check_folded_step(lined_lhs_space(4, 4, 8), (4, 4, 8), 64);
}

/// [`register_matmul_folded_step`] with the block too big for the register budget: the same
/// numbers off the rolled body, whose local arrays are indexed at runtime.
#[test]
fn register_matmul_folded_step_rolled() {
    check_folded_step(lined_lhs_space(4, 4, 8), (4, 4, 8), 8);
}

/// The folded step through a promoted block. The rhs lines along `K`, so the block's lines are
/// one cell's partials, collapsed on drain rather than committed per visit; the sum never meets
/// the output between regions, hence [`a_promoted_folded_step_sums_wider_than_its_output`].
#[test]
fn register_matmul_promoted_folded_step() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (4usize, 4usize, 8usize);
    let launcher = lined_lhs_space(m, n, k);

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[N, K]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let dtype = f32::elem_type_native();
    promoted_matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        4,
        4,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        RegisterBlock::new(64),
        Semiring::SUM_PROD,
        dtype,
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(folded_matmul_reference(m, n, k))
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// A half-precision output summing a long folded contraction: `4096` products of one. Summed in
/// its own element the cell stops at `2048`, where one more product falls under half its spacing
/// and rounds away; carried in `f32` across the walk and cast once on drain, it lands `4096`.
///
/// The property a weight stored along `K` needs from a register block, the folded step being
/// the only way such a weight contracts.
#[test]
fn a_promoted_folded_step_sums_wider_than_its_output() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (1usize, 1usize, 4096usize);
    let launcher = lined_lhs_space(m, n, k);
    let out = f16::elem_type_native();
    let acc = f32::elem_type_native();

    let ones = vec![1.0f32; k];
    let (a, _) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(out)
        .custom(ones.clone())
        .generate_with_f32_host_data();
    let (b, _) = TestInput::builder(client.clone(), shape![n, k])
        .dtype(out)
        .custom(ones)
        .generate_with_f32_host_data();
    let c = TestInput::builder(client.clone(), shape![m, n])
        .dtype(out)
        .zeros()
        .generate_without_host_data();

    promoted_matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        4,
        4,
        1,
        TileArgLaunch::new(a.binding().into_tensor_arg(), TileSpec::direct(&[M, K])),
        TileArgLaunch::new(b.binding().into_tensor_arg(), TileSpec::direct(&[N, K])),
        TileArgLaunch::new(
            c.clone().binding().into_tensor_arg(),
            TileSpec::direct(&[M, N]),
        ),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        RegisterBlock::new(64),
        Semiring::SUM_PROD,
        out,
        acc,
    );

    let got = HostData::from_tensor_handle(&client, c, HostDataType::F32);
    let have = got.get_f32(&[0, 0]);
    assert!(
        have == k as f32,
        "an f16 cell summed in f32 holds {k}, got {have}"
    );
}

/// The N-D nest at a folded step: two contracted axes, both operands lined along the faster of
/// them. The reduce nest steps by the served width, so each step lands on a line start.
#[test]
fn register_matmul_folded_step_two_contracted_axes() {
    let client = cubecl::test_device().client();
    let (m, n, k1, k2) = (4usize, 4usize, 2usize, 4usize);
    let k = k1 * k2;
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k1), (K2, k2)]),
            Levels::leaf(&[(M, m), (N, n), (K, k1), (K2, k2)])
                .walk_every(&[M, N, K, K2])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K, K2]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[N, K, K2]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        4,
        4,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        one_cube(),
        None,
        launcher.partitioning().level(0),
        RegisterBlock::new(64),
        Semiring::SUM_PROD,
        f32::elem_type_native(),
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(folded_matmul_reference(m, n, k))
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// A folded step whose lhs is packed `q8s` (4 values per word): the pack factor narrows the
/// *physical* line to one `u32` while the served line stays `pack` wide, so the same walk, gate
/// and fold run with a decode added on the read.
#[test]
fn register_matmul_folded_step_quant_q8() {
    let client = cubecl::test_device().client();
    run_folded_step_quant(client, QuantValue::Q8S, (4, 4, 8), 4);
}

/// The `q4s` twin: eight values per word, the brief's headline packing. Needs a device whose
/// vectors reach the factor, so it skips on WGSL-bound targets.
#[test]
fn register_matmul_folded_step_quant_q4() {
    let client = cubecl::test_device().client();
    run_folded_step_quant(client, QuantValue::Q4S, (4, 4, 16), 4);
}

/// The quantized folded step: the weight lined along `K` in packed `u32` words, the activation
/// lined along `K` in plain lines, and the decode sitting between the read and the `fma`. Checks
/// `C[i,j] = Σ_p q[i,p]·scale[i/bm]·B[j,p]`.
fn run_folded_step_quant(
    client: Client,
    value: QuantValue,
    (m, n, k): (usize, usize, usize),
    bm: usize,
) {
    let scheme = QuantScheme::default()
        .per_block([bm as u8, k as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(value);
    let pack = scheme.num_quants();

    let max_width = client.properties().hardware.max_vector_size;
    if pack > max_width {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max_width}, below {value:?}'s packing factor ({pack})"
        )))
        .enforce();
        return;
    }

    let launcher = lined_lhs_space(m, n, k);
    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[N, K]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    matmul_quant_lhs_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        pack,
        QuantTileArgLaunch::new(
            a.tile.tensor_arg(1),
            a.scales_binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]),
            scheme,
            DequantAt::Read,
        ),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        RegisterBlock::new(64),
        u32::elem_type_native(),
        f32::elem_type_native(),
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| (a.q[i * k + p] as f32) * a.scale_values[i / bm] * ((j * k + p) as f32))
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

// ---- the sub-plane fold: UnitShare::Group --------------------------------------

/// The nest a rows-in-flight gemv cuts: the plane splits into aligned groups of `group_units`,
/// each group owning one output row and its units interleaving `K` between them. Every unit holds
/// a *partial* of the row, so the drain is a segmented reduction: `UnitShare::Group`, not `Plane`.
///
/// `groups == 1` is the same nest at `UnitShare::Plane`, which is the case already covered; the
/// point here is a plane carrying several cells at once.
fn unit_group_fold_space(plane_size: usize, group_units: usize, edge: usize, n: usize) -> Launcher {
    let groups = plane_size / group_units;
    implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, groups), (N, n), (K, group_units * edge)]),
            Levels::leaf(&[(M, 1), (K, edge)])
                .units(&[(M, groups), (K, group_units)])
                .interleaved(K)
                .build(),
        ),
        Form::Static,
    )
}

/// The memory-backed leaf over the segmented fold, the control for
/// [`register_matmul_promoted_unit_group_fold`]. If this one fails the space itself is wrong and
/// the promoted result proves nothing.
#[test]
fn register_matmul_unit_group_fold() {
    let client = cubecl::test_device().client();
    let plane_units = client.properties().hardware.plane_size_max as usize;
    let (group_units, edge, n) = (8usize, 4usize, 1usize);
    // A fold over unit groups needs a plane that holds at least one whole group.
    if skip_unless_plane_holds(&client, group_units as u32) {
        return;
    }
    let (groups, k) = (plane_units / group_units, group_units * edge);
    let m = groups;
    let launcher = unit_group_fold_space(plane_units, group_units, edge, n);

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        one_cube(),
        None,
        launcher.partitioning().level(0),
        RegisterBlock::new(edge * n),
        Semiring::SUM_PROD,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// The same segmented fold through a **promoted** accumulator.
///
/// The case no other test covers: every promoted test here folds either nothing (`Whole`) or the
/// whole plane (`Plane`). A plane carrying one cell per group has to reduce within each group and
/// let each group's first unit write *its own row*.
///
/// The rows a group owns are what the `M` cut hands it, which a block built before the walk
/// descends has to be told rather than assume.
#[test]
fn register_matmul_promoted_unit_group_fold() {
    let client = cubecl::test_device().client();
    let plane_units = client.properties().hardware.plane_size_max as usize;
    let (group_units, edge, n) = (8usize, 4usize, 1usize);
    // A fold over unit groups needs a plane that holds at least one whole group.
    if skip_unless_plane_holds(&client, group_units as u32) {
        return;
    }
    let (groups, k) = (plane_units / group_units, group_units * edge);
    let (m, dtype) = (groups, f32::elem_type_native());
    let launcher = unit_group_fold_space(plane_units, group_units, edge, n);

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    promoted_matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        RegisterBlock::new(edge * n),
        Semiring::SUM_PROD,
        dtype,
        dtype,
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// The gemv's own layout, promoted: a plane split into groups, each owning a row, its units
/// interleaving `K`, and the weight stored along `K` lining both operands along the contraction.
/// Every unit's block is one line of partials, so the drain folds line then group and writes once.
///
/// The memory-backed leaf does both folds per visit and rounds the cell at each; a half-precision
/// cell rounds away the walk that way, so this is the block a half-precision gemv runs in.
#[test]
fn register_matmul_promoted_folded_step_unit_group_fold() {
    let client = cubecl::test_device().client();
    let plane_units = client.properties().hardware.plane_size_max as usize;
    let (group_units, edge, n) = (8usize, 4usize, 1usize);
    // A fold over unit groups needs a plane that holds at least one whole group.
    if skip_unless_plane_holds(&client, group_units as u32) {
        return;
    }
    let (groups, k) = (plane_units / group_units, group_units * edge);
    let (m, dtype) = (groups, f32::elem_type_native());
    let launcher = unit_group_fold_space(plane_units, group_units, edge, n);

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[N, K]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    promoted_matmul_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        edge,
        edge,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        RegisterBlock::new(edge * n),
        Semiring::SUM_PROD,
        dtype,
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(folded_matmul_reference(m, n, k))
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// A packed lhs contracts into a register block to the product its scales and values describe.
///
/// The decode belongs to the read, not to the leaf: `Tile::matrix_packed` dequantizes per read
/// for whichever leaf asks, so a promoted accumulator serves a quantized operand with nothing of
/// its own. The reference is built on the host from the values and scales, so it shares no decode.
#[test]
fn register_matmul_promoted_accumulator_quant() {
    let client = cubecl::test_device().client();
    let (m, n, k, edge, bm) = (4usize, 4usize, 8usize, 4usize, 4usize);
    let scheme = QuantScheme::default()
        .per_block([bm as u8, k as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(QuantValue::Q8S);
    let pack = scheme.num_quants();

    let max_width = client.properties().hardware.max_vector_size;
    if pack > max_width {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max_width}, below the packing factor ({pack})"
        )))
        .enforce();
        return;
    }

    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    // Poisoned: the kernel owns `out = A·B` whatever the buffer held.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    promoted_matmul_quant_lhs_in_place::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        QuantTileArgLaunch::new(
            a.tile.tensor_arg(1),
            a.scales_binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]),
            scheme,
            DequantAt::Read,
        ),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        RegisterBlock::new(64),
        u32::elem_type_native(),
        f32::elem_type_native(),
        f32::elem_type_native(),
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // Row-major arange rhs: b(p, j) = p·n + j.
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| (a.q[i * k + p] as f32) * a.scale_values[i / bm] * ((p * n + j) as f32))
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

// ---- cmma fragment transit (tensor-core) -------------------------------------

/// Round-trips a 16×16 tile through a tensor-core *accumulator* fragment with no arithmetic,
/// gmem → smem → cmma (load) → smem → gmem, to check that the `TileKind::Cmma` transit
/// (`cmma::load_with_layout` / `cmma::store`) preserves data. Tensor-core only: `cargo test-metal`.
#[test]
fn cmma_fragment_roundtrip() {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let dtype = f32::elem_type_native();
    let space = Space::new(&[(M, 8), (N, 8)]);

    let input = TileInput::builder(&client, space.clone())
        .untiled()
        .arange();
    let output = TileInput::builder(&client, space.clone()).untiled().zeros();

    cmma_roundtrip::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        input.arg(),
        output.arg(),
        uncut(&client, &space, &space).partitioning_arg(),
        dtype,
    );

    let got = HostData::from_tensor_handle(&client, output.handle(), HostDataType::F32);
    let want = HostData::from_tensor_handle(&client, input.handle(), HostDataType::F32);
    assert_equals_approx(&got, &want, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// A real 8×8×8 matmul through tensor cores: `C = A · B`, contracted by `cmma::execute`
/// on the cmma final space. Validates the fragment load → `execute` → store path against
/// the register reference. Tensor-core only: run with `cargo test-metal`.
#[test]
fn cmma_matmul_8x8x8() {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let dtype = f32::elem_type_native();
    let space = Space::new(&[(M, 8), (N, 8), (K, 8)]);
    let a = TileInput::builder(&client, space.subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.subspace(&[M, N]))
        .untiled()
        .zeros();

    cmma_matmul::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        a.arg(),
        b.arg(),
        c.arg(),
        uncut(&client, &space, &space).partitioning_arg(),
        dtype,
    );
    assert_matmul_arange(&client, c.handle(), 8, 8, 8);
}

/// `C = A · Bᵀ` where `B` is stored `{N, K}` row-major: the rhs fragment states
/// [`ColMajor`](MatrixLayout::ColMajor) and reads the same buffer at the same row stride, the
/// score matmul of attention (`Q · Kᵀ`, `K` stored `{S, D}`) in miniature. Tensor-core only.
#[test]
fn cmma_matmul_transposed_rhs_8x8x8() {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let dtype = f32::elem_type_native();
    let space = Space::new(&[(M, 8), (N, 8), (K, 8)]);
    let a = TileInput::builder(&client, space.subspace(&[M, K]))
        .untiled()
        .arange();
    // `{N, K}`: the rhs transposed, so `b[j, p] = j·8 + p`.
    let b = TileInput::builder(&client, space.subspace(&[N, K]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.subspace(&[M, N]))
        .untiled()
        .zeros();

    cmma_matmul_transposed_rhs::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        a.arg(),
        b.arg(),
        c.arg(),
        uncut(&client, &space, &space).partitioning_arg(),
        dtype,
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let (_, expected) = TestInput::builder(client, shape![8, 8])
        .custom(folded_matmul_reference(8, 8, 8))
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Quantized `A` (i8, `scheme`) through the hand-written cmma matmul: `A` dequantizes into smem,
/// then the tensor-core matmul runs in f32. `C[i, j] = Σ_p a[i, p]·scale(i, p)·(p·8 + j)`, the
/// scale block `scales` describes.
fn check_cmma_matmul_quant_8x8x8(
    scheme: QuantScheme,
    scales_shape: (usize, usize),
    scale_vals: Vec<f32>,
    scale_of: impl Fn(usize, usize) -> usize,
) {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) || !require_native_i8(&client) {
        return;
    }

    let a_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![8, 8])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let scales = TestInput::builder(client.clone(), shape![scales_shape.0, scales_shape.1])
        .custom(scale_vals.clone())
        .generate_without_host_data();

    // B: f32 row-major arange (b[p, j] = p·8 + j); C: zeros.
    let b = TileInput::builder(&client, Space::new(&[(K, 8), (N, 8)]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, Space::new(&[(M, 8), (N, 8)]))
        .untiled()
        .zeros();

    let space = Space::new(&[(M, 8), (N, 8), (K, 8)]);
    cmma_matmul_quant::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        QuantTileArgLaunch::new(
            a_input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]),
            scheme,
            DequantAt::Load,
        ),
        b.arg(),
        c.arg(),
        uncut(&client, &space, &space).partitioning_arg(),
        a_dtype,
        f32::elem_type_native(),
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected: Vec<f32> = (0..8 * 8)
        .map(|idx| {
            let (i, j) = (idx / 8, idx % 8);
            (0..8)
                .map(|p| {
                    (a_host.get_f32(&[i, p]) * scale_vals[scale_of(i, p)]) * ((p * 8 + j) as f32)
                })
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![8, 8])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// Per-tensor-quantized `A` (i8) through the cmma matmul. Needs both cmma and native i8.
#[test]
fn cmma_matmul_quant_per_tensor_8x8x8() {
    let scheme = QuantScheme::default()
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);
    check_cmma_matmul_quant_8x8x8(scheme, (1, 1), vec![0.05], |_, _| 0);
}

/// Block-quantized `A` (block along `M`): one flat `8×8` smem fill spans both scale blocks, the
/// per-line lookup picking each line's scale: `A`'s space needs no block sub-level. The cmma
/// fragment then reads the whole `8×8` smem. Validates block windowing into the matmul stage.
#[test]
fn cmma_matmul_quant_block_m_8x8x8() {
    let bm = 4usize; // 2 blocks along M, each 4×8; one scale each
    let scheme = QuantScheme::default()
        .per_block([bm as u8, 8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);
    let scale_vals: Vec<f32> = (0..8 / bm).map(|k| 0.05 * (k + 1) as f32).collect();
    check_cmma_matmul_quant_8x8x8(scheme, (8 / bm, 1), scale_vals, move |i, _| i / bm);
}

/// Block-quantized `A` along `K` (the contraction axis): the scale changes partway through each
/// dot product, and the per-line lookup picks the right one mid-row. The case that matters for
/// quantized-weight matmul.
#[test]
fn cmma_matmul_quant_block_k_8x8x8() {
    let bk = 4usize; // 2 blocks along K, each 8×4; the scale changes at p = 4
    let scheme = QuantScheme::default()
        .per_block([8, bk as u8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);
    let scale_vals: Vec<f32> = (0..8 / bk).map(|k| 0.05 * (k + 1) as f32).collect();
    check_cmma_matmul_quant_8x8x8(scheme, (1, 8 / bk), scale_vals, move |_, p| p / bk);
}

// ---- the cmma K walk ----------------------------------------------------------------

/// A matmul through tensor cores with a K walk: the kernel opens the accumulator fragment, the
/// staged K regions accumulate into it, and the copy back to gmem is the epilogue. Tensor-core
/// only: run with `cargo test-metal`.
#[test]
fn cmma_matmul_staged_k_walk() {
    check_cmma_matmul_k_walk(16, 1, 1, StageLayout::Tiled);
}

/// The double-buffered variant: four K regions rotating through two smem slots, the
/// accumulator fragment resident across all of them.
#[test]
fn cmma_matmul_double_buffered_k_walk() {
    check_cmma_matmul_k_walk(32, 2, 1, StageLayout::Tiled);
}

/// An odd region total (three K stages): the loop leaves the last region primed in slot 0;
/// the epilogue must publish and consume it.
#[test]
fn cmma_matmul_double_buffered_odd_k_walk() {
    check_cmma_matmul_k_walk(24, 2, 1, StageLayout::Tiled);
}

/// The K walk staged into a plain strided stage (the legacy `sync_full_strided` storage):
/// the cmma window transport reads through the layout stack either way.
#[test]
fn cmma_matmul_staged_k_walk_strided_stage() {
    check_cmma_matmul_k_walk(16, 1, 1, StageLayout::Strided);
}

/// The staged cmma K walk with operands served in 2-wide lines: the cooperative fill
/// moves lines, the cmma transport addresses the scalar buffer underneath.
#[test]
fn cmma_matmul_staged_k_walk_vectorized() {
    check_cmma_matmul_k_walk(16, 1, 2, StageLayout::Tiled);
}

/// The staged cmma K walk with the rhs stored `{N, K}`: the stage keeps that order and the `B`
/// fragment loads col-major off it, so a weight contiguous along the contraction is staged in
/// its own lines. Double-buffered, so the stages' prefetch moves that order too.
#[test]
fn cmma_matmul_double_buffered_k_walk_transposed_rhs() {
    check_cmma_matmul_k_walk_with(32, 2, 1, StageLayout::Tiled, true);
}

/// The one level always stages, whatever its depth: a cmma leaf cannot consume the global inputs
/// directly, so the kernel first materializes them in shared memory.
fn check_cmma_matmul_k_walk(k: usize, depth: usize, v: usize, layout: StageLayout) {
    check_cmma_matmul_k_walk_with(k, depth, v, layout, false)
}

/// [`check_cmma_matmul_k_walk`], the rhs stored `{K, N}` or — `transposed` — `{N, K}`, against
/// the reference each storage order states over the same arange.
fn check_cmma_matmul_k_walk_with(
    k: usize,
    depth: usize,
    v: usize,
    layout: StageLayout,
    transposed: bool,
) {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, edge) = (8usize, 8usize, 8usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    let storage = layout.storage(&launcher);

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b_axes: &[Axis] = if transposed { &[N, K] } else { &[K, N] };
    let b = TileInput::builder(&client, launcher.space().subspace(b_axes))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel zeroes the accumulator fragment.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    cmma_matmul_k_walk::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        v,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        storage,
        depth,
        f32::elem_type_native(),
    );
    if transposed {
        let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
        let (_, expected) = TestInput::builder(client, shape![m, n])
            .custom(folded_matmul_reference(m, n, k))
            .generate_with_f32_host_data();
        assert_equals_approx(&output, &expected, 1e-3)
            .as_test_outcome()
            .enforce()
    } else {
        assert_matmul_arange(&client, c.handle(), m, n, k);
    }
}

/// [`cmma_matmul_k_walk`] with the instruction stated rather than picked: one body, launched at
/// a register block and at cmma.
///
/// The whole of the difference between the two is the argument. Nothing below matches on it —
/// [`Tile::accumulator`] opens what the plane sums into, and `mma` reads the staged operands the
/// way that accumulator wants them.
#[cube(launch)]
fn staged_matmul_on_a_stated_instruction<E: Numeric, V: Size>(
    a: &TileArg<'_, E, V>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    space: Partitioning,
    #[comptime] level: Level,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
    #[comptime] instruction: Instruction,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));
    let mut acc = c.accumulator::<E, E, E>(&a, &b, instruction, Monoid::Sum);
    acc.zero();
    let walk = space.over(&level);
    let mut stages = Stages::smem(&walk, &a, &b, storage, depth);
    stages.pipelined(walk, |slot, region| {
        let mut acc_r = acc.at(region);
        slot.consume(|a_s, b_s| {
            acc_r.mma(a_s, b_s, Semiring::SUM_PROD);
        });
    });
    for r0 in c.over(&level).unrolled() {
        let mut c_w = c.at(&r0);
        c_w.copy_cast_from(&acc.at(&r0));
    }
}

/// The staged body at a register block, which every device runs.
#[test]
fn one_staged_body_at_a_register_block() {
    check_staged_matmul_on_a_stated_instruction(Instruction::Registers {
        config: REGISTER_BLOCK,
    });
}

/// The same body at cmma, on a device that offers `8x8x8`.
///
/// The pair of tests is the claim the stated instruction exists for: a kernel whose form is
/// *data* is written once. Two tests rather than one loop, so a device without tensor cores
/// reports the cmma leg as skipped instead of quietly running half the claim.
#[test]
fn one_staged_body_at_cmma() {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }
    check_staged_matmul_on_a_stated_instruction(Instruction::Cmma);
}

/// [`staged_matmul_on_a_stated_instruction`] against the arange reference, at whichever form.
fn check_staged_matmul_on_a_stated_instruction(instruction: Instruction) {
    let client = cubecl::test_device().client();
    let (m, n, k, edge) = (8usize, 8usize, 16usize, 8usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    staged_matmul_on_a_stated_instruction::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        StageLayout::Tiled.storage(&launcher),
        1,
        instruction,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// The raw-mma twin of `cmma_matmul_staged_k_walk`: the same open → zero → mma → drain kernel,
/// but the contraction runs through `MmaDefinition::execute` over register fragments rather than
/// the cooperative `cmma::execute`.
///
/// Gated on the backend exposing the manual-mma feature (`features.matmul.mma`); uses the
/// universal manual transport (`MmaIo::manual()`), so no `ldmatrix`/`stmatrix` path is
/// taken. Run with `cargo test-metal` / `test-cuda` on a backend that advertises manual mma.
#[test]
fn mma_matmul_8x8x8() {
    let client = cubecl::test_device().client();
    if !require_mma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, k, edge) = (8usize, 8usize, 8usize, 8usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel zeroes the accumulator fragment.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    mma_matmul_k_walk::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        MmaIo::manual(),
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// The multi-plane cmma stage: a double-buffered K walk fills a shared `16×8`/`8×16` stage
/// cooperatively (cyclic across the cube's 128 units), and a plane-partitioned inner level hands
/// each of the 4 planes an `8×8` fragment resident across all four K steps. Tensor-core only.
#[test]
fn cmma_matmul_plane_partitioned_stage() {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, k, edge) = (16usize, 16usize, 32usize, 8usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .planes(&[(M, m / edge), (N, n / edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel zeroes the accumulator fragment.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    cmma_matmul_two_levels_planes::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        2,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// The multi-fragment partition: each of the 4 planes owns a 2×2 partition of 8³ fragments,
/// resident across a double-buffered K walk; the fragment level reads the stage where it lies, so
/// the unrolled walk reloads operand fragments per execute (no restaging). Tensor-core only.
#[test]
fn cmma_matmul_multi_fragment_partition() {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, k) = (32usize, 32usize, 32usize);
    let (part, i, stage_k) = (16usize, 8usize, 16usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, i), (N, i), (K, i)])
                .walk(&[(M, part / i), (N, part / i), (K, stage_k / i)])
                .planes(&[(M, m / part), (N, n / part)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel zeroes the accumulator fragments.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    cmma_matmul_three_levels_planes_fragments::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        launcher.partitioning().level(2),
        2,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// The legacy register budget as a level structure: a contraction-step walk (windowing only), an
/// N-walk loading one B fragment per step while the A column loads once above it, and an M-only
/// fragment walk below: sub-block partition selection and unrolled fragment walks. Tensor-core only
#[test]
fn cmma_matmul_staged_n_walk_partition() {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, k) = (32usize, 32usize, 32usize);
    let (part, i, stage_k) = (16usize, 8usize, 16usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, i), (N, i), (K, i)])
                .walk(&[(M, part / i), (N, 1), (K, 1)])
                .walk(&[(M, 1), (N, part / i), (K, 1)])
                .walk(&[(M, 1), (N, 1), (K, stage_k / i)])
                .planes(&[(M, m / part), (N, n / part)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a = TileInput::builder(&client, launcher.space().subspace(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel zeroes the accumulator fragments.
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    cmma_matmul_five_levels::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a.arg(),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        launcher.partitioning().level(1),
        launcher.partitioning().level(2),
        launcher.partitioning().level(3),
        launcher.partitioning().level(4),
        2,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

// ---- the quantized cmma K walk ----------------------------------------------------

/// Per-tensor-quantized `A` (i8) through the K walk, staged: `K = 16` runs in two
/// `8`-deep K regions, and each region's smem fill dequantizes `A` on its own. The
/// self-describing fill in action. Tensor-core only.
#[test]
fn cmma_matmul_quant_k_walk() {
    check_cmma_matmul_quant_k_walk(16, 1);
}

/// The same self-describing quant K walk driven double-buffered: both slots' fills dequantize.
#[test]
fn cmma_matmul_quant_double_buffered_k_walk() {
    check_cmma_matmul_quant_k_walk(32, 2);
}

fn check_cmma_matmul_quant_k_walk(k: usize, depth: usize) {
    let scheme = QuantScheme::default()
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);
    check_cmma_matmul_quant_walk(k, depth, 1, scheme, (1, 1), vec![0.05], |_, _| 0);
}

/// Block-M-quantized `A` through the K walk: one K stage stages the whole `M = 8`, which spans
/// two `bm = 4` scale blocks, so a single cooperative fill dequantizes across two scales: the
/// per-line scale lookup, not the one-scale-per-window assumption. Tensor-core only.
#[test]
fn cmma_matmul_quant_block_m_k_walk() {
    let (m, k, bm) = (8usize, 16usize, 4usize); // 2 M-blocks
    let scheme = QuantScheme::default()
        .per_block([bm as u8, k as u8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);
    let scale_vals: Vec<f32> = (0..m / bm).map(|b| 0.05 * (b + 1) as f32).collect();
    check_cmma_matmul_quant_walk(k, 1, 1, scheme, (m / bm, 1), scale_vals, move |i, _| i / bm);
}

/// Block-K-quantized `A` through the K walk (the quantized-weight case): the scale changes
/// partway through each `8`-deep K stage (`bk = 4`), and it changes again between stages, so
/// the per-line scale lookup must fold in the stage's `window_start`. Tensor-core only.
#[test]
fn cmma_matmul_quant_block_k_k_walk() {
    let (m, k, bk) = (8usize, 16usize, 4usize); // 4 K-blocks, 2 per stage
    let scheme = QuantScheme::default()
        .per_block([m as u8, bk as u8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);
    let scale_vals: Vec<f32> = (0..k / bk).map(|b| 0.05 * (b + 1) as f32).collect();
    check_cmma_matmul_quant_walk(k, 1, 1, scheme, (1, k / bk), scale_vals, move |_, p| p / bk);
}

/// Block-K-quantized `A` served in 2-wide lines: the blocks sit on the vectorized inner axis, so
/// a line's coordinate counts lines while its scale block is cut in elements, the widening
/// [`ScaleLayout`] does. Two lines per `bk = 4` block, so a scale changes mid-fill. Tensor-core.
#[test]
fn cmma_matmul_quant_block_k_k_walk_vectorized() {
    let (m, k, bk) = (8usize, 16usize, 4usize);
    let scheme = QuantScheme::default()
        .per_block([m as u8, bk as u8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);
    let scale_vals: Vec<f32> = (0..k / bk).map(|b| 0.05 * (b + 1) as f32).collect();
    check_cmma_matmul_quant_walk(k, 1, 2, scheme, (1, k / bk), scale_vals, move |_, p| p / bk);
}

/// Drive [`cmma_matmul_k_walk_quant`] over `8 × 8 × k` in `8`-deep stages and check
/// `C[i, j] = Σ_p (a[i, p] · scale(i, p)) · (p·n + j)`.
fn check_cmma_matmul_quant_walk(
    k: usize,
    depth: usize,
    v: usize,
    scheme: QuantScheme,
    scales_shape: (usize, usize),
    scale_vals: Vec<f32>,
    scale_of: impl Fn(usize, usize) -> usize,
) {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) || !require_native_i8(&client) {
        return;
    }

    let (m, n, edge) = (8usize, 8usize, 8usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let a_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let scales = TestInput::builder(client.clone(), shape![scales_shape.0, scales_shape.1])
        .custom(scale_vals.clone())
        .generate_without_host_data();

    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .zeros();

    cmma_matmul_k_walk_quant::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        v,
        QuantTileArgLaunch::new(
            a_input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]),
            scheme,
            DequantAt::Load,
        ),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        depth,
        a_dtype,
        f32::elem_type_native(),
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| {
                    (a_host.get_f32(&[i, p]) * scale_vals[scale_of(i, p)]) * ((p * n + j) as f32)
                })
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

/// The manual-mma leaf decoding at the *read*: `DequantAt::Read` keeps `A`'s stage in its stored
/// `i8`, and the fragment load decodes each element through the quant-transparent matrix view,
/// which the cmma twin cannot (its load takes a raw window). Same numbers, a quarter the stage.
#[test]
fn mma_matmul_quant_until_read() {
    let client = cubecl::test_device().client();
    // The shape, not just the feature; see `require_mma_8x8x8_f32`. The `f32` triple, not the
    // stored `i8` one, and `8x8x8`, not `8x8x16`: `K = 16` is the *walk*, walked 8 deep, and `A`
    // decodes at the read, so the leaf uses the same f32 `8x8x8` the plain manual-mma test runs.
    if !require_mma_8x8x8_f32(&client) || !require_native_i8(&client) {
        return;
    }

    let (m, n, k, edge) = (8usize, 8usize, 16usize, 8usize);
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, m), (N, n), (K, k)]),
            Levels::leaf(&[(M, edge), (N, edge), (K, edge)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );

    let scale = 0.05f32;
    let scheme = QuantScheme::default()
        .per_tensor(ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    let a_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let scales = TestInput::builder(client.clone(), shape![1, 1])
        .custom(vec![scale])
        .generate_without_host_data();

    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .zeros();

    mma_matmul_k_walk_quant::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        QuantTileArgLaunch::new(
            a_input.binding().into_tensor_arg(),
            scales.binding().into_tensor_arg(),
            None.into(),
            None.into(),
            TileSpec::direct(&[M, K]),
            scheme,
            DequantAt::Read,
        ),
        b.arg(),
        c.arg(),
        launcher.partitioning_arg(),
        launcher.partitioning().level(0),
        MmaIo::manual(),
        a_dtype,
        f32::elem_type_native(),
    );

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| (a_host.get_f32(&[i, p]) * scale) * ((p * n + j) as f32))
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

// ---- Quantized A through the register (plain-ALU) leaf --------------------------------
//
// Every other quant matmul above runs on tensor cores and skips where cmma is absent, which is
// everywhere the memory-bound GEMV actually lives. These pin the other leaf.
//
// The staged kernel stages `A`'s *packed storage words* into smem, and the software instruction
// dequantizes each read out of smem: no f32 inflation of the stage, no promotion, no cmma, no
// i8 needed for the packed cases (the binding is a `u32`).

/// Native i8 `A`, one scale per `bm`-row block, through the register leaf.
#[test]
fn register_matmul_quant_native_block_m() {
    run_register_matmul_quant_native(Serve::Staged);
}

/// Native i8 `A` served DIRECTLY through the register leaf (Keystone K): nothing is staged, so
/// the leaf reads i8 straight from gmem and scales per read. The native + lhs-arm twin of the
/// packed-rhs [`register_matmul_quant_rhs_direct_serve_gemv`]; the pair covers the quant dispatch.
#[test]
fn register_matmul_quant_native_direct_serve() {
    run_register_matmul_quant_native(Serve::Direct);
}

fn run_register_matmul_quant_native(serve: Serve) {
    let client = cubecl::test_device().client();
    if !require_native_i8(&client) {
        return;
    }

    let (m, n, k, bm) = (8usize, 8usize, 8usize, 4usize);
    let scheme = QuantScheme::default()
        .per_block([bm as u8, k as u8], ScaleDtype::F32)
        .with_store(QuantStore::Native)
        .with_value(QuantValue::Q8S);

    let a_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let q: Vec<f32> = (0..m * k)
        .map(|idx| a_host.get_f32(&[idx / k, idx % k]))
        .collect();

    let scale_vals: Vec<f32> = (0..m / bm).map(|g| 0.05 * (g + 1) as f32).collect();
    let scales = TestInput::builder(client.clone(), shape![m / bm, 1])
        .custom(scale_vals.clone())
        .generate_without_host_data();

    run_register_matmul_quant(
        client,
        (m, n, k),
        Levels::leaf(&[(M, 4), (N, 4), (K, 4)]).walk_every(&[M, N, K]),
        serve,
        a_input.binding().into_tensor_arg(),
        a_dtype,
        scheme,
        scales.binding().into_tensor_arg(),
        scale_vals,
        bm,
        q,
    );
}

/// Packed-u32 Q8S `A` (4 values per word along `K`), served in whole-word lines.
#[test]
fn register_matmul_quant_packed_q8() {
    let client = cubecl::test_device().client();
    run_register_matmul_quant_packed(client, (8, 8, 8), 4, QuantValue::Q8S, 4);
}

/// Packed-u32 Q4S `A` (8 values per word): the widest served line, so it needs a device
/// whose vectors reach the packing factor (cpu/cuda; WGSL-bound targets cap at 4).
#[test]
fn register_matmul_quant_packed_q4() {
    let client = cubecl::test_device().client();
    run_register_matmul_quant_packed(client, (8, 8, 16), 8, QuantValue::Q4S, 4);
}

/// Build a packed `A` spanning the scheme's signed range and run the register matmul.
fn run_register_matmul_quant_packed(
    client: Client,
    (m, n, k): (usize, usize, usize),
    tk: usize,
    value: QuantValue,
    bm: usize,
) {
    let scheme = QuantScheme::default()
        .per_block([bm as u8, k as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(value);
    let pack = scheme.num_quants();

    let max_width = client.properties().hardware.max_vector_size;
    if pack > max_width {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max_width}, below {value:?}'s packing factor ({pack})"
        )))
        .enforce();
        return;
    }

    let a = TileInput::builder(&client, Space::new(&[(M, m), (K, k)]))
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .arange();

    let a_dtype = u32::elem_type_native();
    let q: Vec<f32> = a.q.iter().map(|&v| v as f32).collect();
    run_register_matmul_quant(
        client,
        (m, n, k),
        Levels::leaf(&[(M, 4), (N, 4), (K, tk)]).walk_every(&[M, N, K]),
        Serve::Staged,
        a.tile.tensor_arg(1),
        a_dtype,
        scheme,
        a.scales_binding().into_tensor_arg(),
        a.scale_values.clone(),
        bm,
        q,
    );
}

/// Drive the quantized-lhs register kernels and check `C[i,j] = Σ_p q[i,p]·scale[i/bm]·B[p,j]`.
/// [`Serve::Staged`] stages `A`'s storage into smem and dequantizes per read out of it,
/// [`Serve::Direct`] serves it from gmem; either way through `matrix_transparent`, no f32 stage.
#[allow(clippy::too_many_arguments)]
fn run_register_matmul_quant(
    client: Client,
    (m, n, k): (usize, usize, usize),
    plan: Levels,
    serve: Serve,
    a_arg: TensorArg,
    a_dtype: ElemType,
    scheme: QuantScheme,
    scales_arg: TensorArg,
    scale_vals: Vec<f32>,
    bm: usize,
    q: Vec<f32>,
) {
    let launcher = implied(
        &client,
        Partitioning::new(Space::new(&[(M, m), (N, n), (K, k)]), plan.build()),
        Form::Static,
    );

    let b = TileInput::builder(&client, launcher.space().subspace(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, launcher.space().subspace(&[M, N]))
        .untiled()
        .zeros();
    let e_dtype = f32::elem_type_native();
    // Built in each arm: a launch argument is typed by the kernel that takes it.
    match serve {
        Serve::Staged => matmul_quant_lhs_smem_ring::launch(
            &client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            QuantTileArgLaunch::new(
                a_arg,
                scales_arg,
                None.into(),
                None.into(),
                TileSpec::direct(&[M, K]),
                scheme,
                DequantAt::Load,
            ),
            b.arg(),
            c.arg(),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            REGISTER_BLOCK,
            a_dtype,
            e_dtype,
        ),
        Serve::Direct => matmul_quant_lhs_in_place::launch(
            &client,
            CubeCount::new_single(),
            CubeDim::new_single(),
            1,
            QuantTileArgLaunch::new(
                a_arg,
                scales_arg,
                None.into(),
                None.into(),
                TileSpec::direct(&[M, K]),
                scheme,
                DequantAt::Load,
            ),
            b.arg(),
            c.arg(),
            launcher.partitioning_arg(),
            launcher.partitioning().level(0),
            REGISTER_BLOCK,
            a_dtype,
            e_dtype,
        ),
    }

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| q[i * k + p] * scale_vals[i / bm] * ((p * n + j) as f32))
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}

// ---- Quantized B (RHS) through the register leaf ---------------------------------------
//
// The gemv production shape: the *weight* is the streamed RHS at `(K, N) = (d_in, d_out)`, packed
// along `d_out` (the innermost axis) with one scale per `(k, N-group)` block (`[1, bn]`); A stays
// float. The RHS's served width drives the accumulator's line width, so `C` launches at that width.

/// Packed-u32 Q8S `B` (4 values per word along `N`), scales `[1, bn]`: the exact scheme
/// family `metabolic`'s gemv ships (`q8s`, packed-u32, block scales along `d_out`).
#[test]
fn register_matmul_quant_rhs_packed_q8() {
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 8), (N, 8), (K, 8)]),
            Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    run_register_matmul_quant_rhs(
        client,
        launcher.clone(),
        QuantValue::Q8S,
        4,
        DequantAt::Read,
        Serve::Staged,
        None,
    );
}

/// The `q4s` twin (8 values per word): needs 8-wide bindings, so cpu/cuda only.
#[test]
fn register_matmul_quant_rhs_packed_q4() {
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 8), (N, 16), (K, 8)]),
            Levels::leaf(&[(M, 4), (N, 8), (K, 4)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    run_register_matmul_quant_rhs(
        client,
        launcher.clone(),
        QuantValue::Q4S,
        8,
        DequantAt::Read,
        Serve::Staged,
        None,
    );
}

/// The decode shape itself: a single activation row (`m = 1`) against the packed weight,
/// what every projection degenerates to during token-by-token generation.
#[test]
fn register_matmul_quant_rhs_gemv_row() {
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 1), (N, 8), (K, 8)]),
            Levels::leaf(&[(M, 1), (N, 4), (K, 4)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    run_register_matmul_quant_rhs(
        client,
        launcher.clone(),
        QuantValue::Q8S,
        4,
        DequantAt::Read,
        Serve::Staged,
        None,
    );
}

/// The decode shape spread across the device: `N` across cubes on `X`, the geometry a gemv
/// selector emits (`M = 1` leaves nothing else to spread).
#[test]
fn register_matmul_quant_rhs_gemv_row_multi_cube() {
    let client = cubecl::test_device().client();
    let launcher = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 1), (N, 16), (K, 8)]),
            Levels::leaf(&[(M, 1), (N, 4), (K, 4)])
                .walk_every(&[M, K])
                .cubes(&[N])
                .build(),
        ),
        Form::Static,
    );
    run_register_matmul_quant_rhs(
        client,
        launcher.clone(),
        QuantValue::Q8S,
        4,
        DequantAt::Read,
        Serve::Staged,
        None,
    );
}

/// Direct-serve the quantized RHS weight (Keystone K): nothing is staged, so the register leaf
/// reads the packed weight straight from gmem and dequantizes *per read* through
/// [`matrix_transparent`]: the sync-free `m = 1` decode path.
///
/// The `_rhs_*` tests above are all staged: they stage the weight's *packed words* into smem
/// (plus its scales) and dequantize per read out of smem. Same answer; direct avoids even the
/// smem round-trip.
#[test]
fn register_matmul_quant_rhs_direct_serve_gemv() {
    let client = cubecl::test_device().client();
    let launch = implied(
        &client,
        Partitioning::new(
            Space::new(&[(M, 1), (N, 8), (K, 8)]),
            Levels::leaf(&[(M, 1), (N, 4), (K, 4)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    );
    let launcher = implied(
        &client,
        Partitioning::new(
            launch.space().clone(),
            launch.partitioning().levels().to_vec(),
        ),
        Form::Dynamic,
    );
    run_register_matmul_quant_rhs(
        client,
        launcher.clone(),
        QuantValue::Q8S,
        4,
        DequantAt::Read,
        Serve::Direct,
        None,
    );
}

/// The Goal path: a staged packed weight whose smem stage holds the *packed u32 words*, not a
/// dequantized f32 stage. A four-region K-walk (`k = 16`, `tk = 4`) with `[1, bn]` scales distinct
/// along K refills both per region; the leaf decodes each smem read via [`matrix_transparent`].
///
/// This is the batched weight-streaming case the change targets: the contrast to the f32-inflated
/// stage the cmma leaf still uses, and to the sync-free direct serve above.
#[test]
fn register_matmul_quant_rhs_staged_packed_smem() {
    let client = cubecl::test_device().client();
    run_register_matmul_quant_rhs(
        client,
        four_region_k_walk(),
        QuantValue::Q8S,
        4,
        DequantAt::Read,
        Serve::Staged,
        None,
    );
}

/// The same staged packed weight, decoded by the load instead of the read (`DequantAt::Load`): the
/// stage holds served values, so it costs the served-to-stored ratio in shared memory and decodes
/// once per element rather than per read: the fork a register leaf may take and a cmma leaf must.
#[test]
fn register_matmul_quant_rhs_staged_dequantized_smem() {
    let client = cubecl::test_device().client();
    run_register_matmul_quant_rhs(
        client,
        four_region_k_walk(),
        QuantValue::Q8S,
        4,
        DequantAt::Load,
        Serve::Staged,
        None,
    );
}

/// Two-level through the staged `DequantAt::Read` path: the stage keeps the packed weight, and
/// `stage_scales` writes `global * local` into the smem scale grid, so reads see one-level
/// scales. The expectation carries the global scale, so a missed or doubled fold fails by it.
#[test]
fn register_matmul_quant_rhs_two_level_staged_packed_smem() {
    let client = cubecl::test_device().client();
    run_register_matmul_quant_rhs(
        client,
        four_region_k_walk(),
        QuantValue::Q8S,
        4,
        DequantAt::Read,
        Serve::Staged,
        Some(0.5),
    );
}

/// Two-level through the staged `DequantAt::Load` path: the fill dequantizes into the stage, so
/// the global scale folds in the gmem read itself and the stage carries plain served values.
#[test]
fn register_matmul_quant_rhs_two_level_staged_dequantized_smem() {
    let client = cubecl::test_device().client();
    run_register_matmul_quant_rhs(
        client,
        four_region_k_walk(),
        QuantValue::Q8S,
        4,
        DequantAt::Load,
        Serve::Staged,
        Some(0.5),
    );
}

/// `4 × 8 × 16` walked in `4×4×4` tiles: four K regions per output tile.
fn four_region_k_walk() -> Launcher {
    implied(
        &cubecl::test_device().client(),
        Partitioning::new(
            Space::new(&[(M, 4), (N, 8), (K, 16)]),
            Levels::leaf(&[(M, 4), (N, 4), (K, 4)])
                .walk_every(&[M, N, K])
                .build(),
        ),
        Form::Static,
    )
}

/// Drive the quantized-rhs register kernels and check
/// `C[i,j] = Σ_p A[i,p] · q_b[p,j] · scale[p, j/bn]`.
#[allow(clippy::too_many_arguments)]
fn run_register_matmul_quant_rhs(
    client: Client,
    launch: Launcher,
    value: QuantValue,
    bn: usize,
    dequant_at: DequantAt,
    serve: Serve,
    global: Option<f32>,
) {
    // The data is minted against the one-level scheme either way: a two-level tensor holds the
    // same value and block-scale bytes, plus the global scale in its own binding.
    let mint_scheme = QuantScheme::default()
        .per_block([1, bn as u8], ScaleDtype::F32)
        .with_store(QuantStore::PackedU32(0))
        .with_value(value);
    let scheme = match global {
        Some(_) => mint_scheme.per_tensor(ScaleDtype::F32),
        None => mint_scheme,
    };
    let pack = scheme.num_quants();

    let max_width = client.properties().hardware.max_vector_size;
    if pack > max_width {
        TestOutcome::Validated(ValidationResult::Skipped(format!(
            "device vectors cap at {max_width}, below {value:?}'s packing factor ({pack})"
        )))
        .enforce();
        return;
    }

    let (m, n, k) = (
        launch.space().extent(M),
        launch.space().extent(N),
        launch.space().extent(K),
    );
    let a = TileInput::builder(&client, launch.space().subspace(&[M, K]))
        .untiled()
        .arange();
    // The weight and its per-(k, N-group) scales, minted together.
    let b = TileInput::builder(&client, launch.space().subspace(&[K, N]))
        .untiled()
        .packed(&mint_scheme, dequant_at)
        .arange();
    let global_scale = global.map(|g| {
        TestInput::builder(client.clone(), shape![1])
            .custom(vec![g])
            .generate_without_host_data()
    });
    let c = TileInput::builder(&client, launch.space().subspace(&[M, N]))
        .untiled()
        .zeros();
    let b_dtype = u32::elem_type_native();
    let e_dtype = f32::elem_type_native();

    // Routine-like: the launcher derives geometry and argument wiring from the nest; the
    // quantized RHS goes through the source builder, which binds it at the storage width.
    let launcher = implied(
        &client,
        Partitioning::new(
            launch.space().clone(),
            launch.partitioning().levels().to_vec(),
        ),
        Form::Dynamic,
    );
    let a_op = launcher.arg(a.handle().binding()).axes(&[M, K]).build();
    let b_op = launcher
        .arg(b.tile.handle().binding())
        .axes(&[K, N])
        .vectorize(pack)
        .quantized(Quantization::new(
            b.scales_binding(),
            global_scale.map(|g| g.binding()),
            scheme,
            dequant_at,
        ))
        .build();
    // The register instruction lines the accumulator at the RHS's served width.
    let c_op = launcher
        .arg(c.handle().binding())
        .axes(&[M, N])
        .vectorize(pack)
        .build();
    // One level cuts `N` across cubes where the test says so; the walk is always stated.
    let (outer, inner) = match launch.partitioning().levels().len() {
        1 => (one_cube(), launch.partitioning().level(0)),
        _ => (
            launch.partitioning().level(0),
            launch.partitioning().level(1),
        ),
    };
    match serve {
        Serve::Staged => matmul_quant_rhs_smem_ring::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            c_op.vector_size,
            a_op.arg(),
            b_op.quant_arg(),
            c_op.arg(),
            launcher.partitioning_arg(),
            outer.clone(),
            inner.clone(),
            REGISTER_BLOCK,
            b_dtype,
            e_dtype,
        ),
        Serve::Direct => matmul_quant_rhs_in_place::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            c_op.vector_size,
            a_op.arg(),
            b_op.quant_arg(),
            c_op.arg(),
            launcher.partitioning_arg(),
            outer.clone(),
            inner.clone(),
            REGISTER_BLOCK,
            b_dtype,
            e_dtype,
        ),
    }

    let output = HostData::from_tensor_handle(&client, c.handle(), HostDataType::F32);
    // A is arange over (m, k): a[i, p] = i·k + p.
    let sn = n / bn;
    let g = global.unwrap_or(1.0);
    let expected: Vec<f32> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|p| {
                    ((i * k + p) as f32)
                        * (b.q[p * n + j] as f32)
                        * b.scale_values[p * sn + j / bn]
                        * g
                })
                .sum()
        })
        .collect();
    let (_, expected) = TestInput::builder(client, shape![m, n])
        .custom(expected)
        .generate_with_f32_host_data();
    assert_equals_approx(&output, &expected, 1e-3)
        .as_test_outcome()
        .enforce()
}
