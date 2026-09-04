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
    assert_equals_approx,
};

use cubek_tile::*;

use super::references;

/// Skip guard for the tensor-core tests in this file, which all hardcode
/// `8x8x8` `f32` fragments (the native Metal simdgroup shape). Checking only
/// that *some* cmma config exists is not enough: drivers accept only the exact
/// fragment shapes they advertise, and an unsupported shape is rejected at
/// compile time. Returns `false` (after enforcing a skip outcome) when the
/// device doesn't advertise the exact configuration.
fn require_cmma_8x8x8_f32(client: &Client) -> bool {
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

/// The manual-mma twin of [`require_cmma_8x8x8_f32`]. The *shape*, not just the
/// feature: a backend can advertise manual mma and offer only `16x16x16`
/// (gfx1151 does), and running `8x8x8` there is an instruction the hardware does
/// not have: it reads back zeros, which looks like a leaf bug and is a missing
/// guard.
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
/// lane fan-out.
const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(16);

/// Where an operand is read from in the kernels that offer both: staged in shared memory, or
/// where it lies in global memory.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Serve {
    Staged,
    Direct,
}

fn sequential(edges: &[(Axis, usize)]) -> Partitioner {
    let dists: Vec<_> = edges
        .iter()
        .map(|&(a, _)| (a, Distribution::Sequential))
        .collect();
    Partitioner::over(ByAxis::new(edges), ByAxis::new(&dists)).level()
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

/// `c = a · b` with every operand read where it lies: one level, the leaf running the software
/// instruction under `config` on each region, folding under `semiring`. `c` owns its init: the
/// semiring's identity, whatever the buffer held.
#[cube(launch)]
fn matmul_in_place<E: Numeric, AV: Size, BV: Size, CV: Size>(
    a: &TileArg<'_, E, AV>,
    b: &TileArg<'_, E, BV>,
    c: &TileArg<'_, E, CV>,
    #[comptime] space: Space,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.init(Monoid::identity::<E>(comptime!(semiring.add())));
    for region in Walk::over(c.op_space(&a, &b)) {
        let mut c_r = c.at(&region);
        c_r.mma_with(&a.at(&region), &b.at(&region), config, semiring);
    }
}

/// `c = a · b` with both operands staged in shared memory per region of the one level, `depth`
/// regions in flight through the ring.
#[cube(launch)]
fn matmul_smem_ring<E: Numeric, V: Size>(
    a: &TileArg<'_, E, V>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    let walk = Walk::over(c.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Strided, depth);
    pipelined(walk, &mut ring, |slot, region| {
        let mut c_r = c.at(region);
        slot.consume(|a_s, b_s| {
            c_r.mma_with(a_s, b_s, REGISTER_BLOCK, Semiring::SUM_PROD);
        });
    });
}

/// [`matmul_smem_ring`] walking its regions last to first.
#[cube(launch)]
fn matmul_smem_ring_reversed<E: Numeric, V: Size>(
    a: &TileArg<'_, E, V>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    let walk = Walk::over(c.op_space(&a, &b)).reversed();
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Strided, depth);
    pipelined(walk, &mut ring, |slot, region| {
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
    #[comptime] space: Space,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(space);
    let walk = Walk::over(c.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Strided, depth);
    pipelined(walk, &mut ring, |slot, region| {
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
    #[comptime] space: Space,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    let walk = Walk::over(c.op_space(&a, &b));
    let mut ring = Ring::smem_single(&walk, &a, StageStorage::Strided, depth);
    pipelined(walk, &mut ring, |slot, region| {
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
    #[comptime] space: Space,
    #[comptime] width: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    let walk = Walk::over(c.op_space(&a, &b));
    let mut ring = Ring::smem_single_at(
        &walk,
        &b,
        StageStorage::Strided,
        comptime!(Some(width)),
        1usize,
    );
    pipelined(walk, &mut ring, |slot, region| {
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
    #[comptime] space: Space,
    #[comptime] width: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    for outer in Walk::over(c.op_space(&a, &b)) {
        let c_o = c.at(&outer);
        let a_o = a.at(&outer);
        let b_o = b.at(&outer);
        let walk = Walk::over(c_o.op_space(&a_o, &b_o));
        let mut ring = Ring::smem_single_at(
            &walk,
            &a_o,
            StageStorage::Strided,
            comptime!(Some(width)),
            1usize,
        );
        pipelined(walk, &mut ring, |slot, region| {
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
    #[comptime] space: Space,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    let walk = Walk::over(c.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, storage, depth);
    pipelined(walk, &mut ring, |slot, region| {
        let c_o = c.at(region);
        slot.consume(|a_s, b_s| {
            for cell in Walk::over(c_o.op_space(a_s, b_s)) {
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
    #[comptime] space: Space,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    let walk = Walk::over(c.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, storage, depth);
    pipelined(walk, &mut ring, |slot, region| {
        let c_o = c.at(region);
        slot.consume(|a_s, b_s| {
            for cell in Walk::over(c_o.op_space(a_s, b_s)).reversed() {
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

/// Two levels, both staging: the inner ring restages each final tile out of the outer stage.
#[cube(launch)]
fn matmul_two_levels_smem_then_smem<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] storage: StageStorage,
    #[comptime] depth_outer: usize,
    #[comptime] depth_inner: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    let walk = Walk::over(c.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, storage, depth_outer);
    pipelined(walk, &mut ring, |slot, region| {
        let c_o = c.at(region);
        slot.consume(|a_s, b_s| {
            let inner = Walk::over(c_o.op_space(a_s, b_s));
            let mut inner_ring = Ring::smem(&inner, a_s, b_s, storage, depth_inner);
            pipelined(inner, &mut inner_ring, |slot, cell| {
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
    #[comptime] space: Space,
    #[comptime] config: RegisterBlock,
    #[comptime] semiring: Semiring,
    #[define(E)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.block_accumulator::<EA, E>(&a, config, comptime!(semiring.add()));
    acc.init(Monoid::identity::<EA>(comptime!(semiring.add())));
    for region in Walk::over(acc.op_space(&a, &b)) {
        let mut acc_r = acc.at(&region);
        acc_r.mma(&a.at(&region), &b.at(&region), semiring);
    }
    acc.drain_cast_into(&mut c);
}

/// [`promoted_matmul_in_place`] over the two-level cube/plane space a real gemm composes: the
/// block is opened per plane region and contracted along the inner K walk.
#[cube(launch)]
fn promoted_matmul_two_levels_in_place<E: Numeric, EA: Numeric, V: Size>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, V>,
    c: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[comptime] config: RegisterBlock,
    #[define(E)] _dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(space);
    for outer in Walk::over(c.op_space(&a, &b)) {
        let mut c_o = c.at(&outer);
        let a_o = a.at(&outer);
        let b_o = b.at(&outer);
        let mut acc = c_o.block_accumulator::<EA, E>(&a_o, config, Monoid::Sum);
        acc.zero();
        for region in Walk::over(acc.op_space(&a_o, &b_o)) {
            let mut acc_r = acc.at(&region);
            acc_r.mma(&a_o.at(&region), &b_o.at(&region), Semiring::SUM_PROD);
        }
        acc.drain_cast_into(&mut c_o);
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
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.block_accumulator::<E, E>(&a, REGISTER_BLOCK, Monoid::Sum);
    acc.zero();
    for outer in Walk::over(acc.op_space(&a, &b)) {
        let acc_o = acc.at(&outer);
        let a_o = a.at(&outer);
        let b_o = b.at(&outer);
        let walk = Walk::over(acc_o.op_space(&a_o, &b_o)).unrolled();
        let mut ring = Ring::smem(&walk, &a_o, &b_o, StageStorage::Strided, 1usize);
        pipelined(walk, &mut ring, |slot, cell| {
            let mut acc_r = acc_o.at(cell);
            slot.consume(|a_s, b_s| {
                acc_r.mma(a_s, b_s, Semiring::SUM_PROD);
            });
        });
    }
    acc.drain_cast_into(&mut c);
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
    #[comptime] space: Space,
    #[comptime] storage: StageStorage,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.cmma_accumulator::<E, E>(&a, Monoid::Sum);
    acc.zero();
    let walk = Walk::over(acc.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, storage, depth);
    pipelined(walk, &mut ring, |slot, region| {
        let mut acc_r = acc.at(region);
        slot.consume(|a_s, b_s| {
            acc_r.mma(a_s, b_s, Semiring::SUM_PROD);
        });
    });
    acc.drain_cast_into(&mut c);
}

/// [`cmma_matmul_k_walk`] with a quantized lhs: each region's stage decodes it (or keeps it stored
/// for the fragment load to decode) exactly as the operand states.
#[cube(launch)]
fn cmma_matmul_k_walk_quant<I: Numeric, E: Numeric, V: Size>(
    a: &QuantTileArg<'_, I, V>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] depth: usize,
    #[define(I)] _idtype: ElemType,
    #[define(E)] _edtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.cmma_accumulator::<E, E>(&a, Monoid::Sum);
    acc.zero();
    let walk = Walk::over(acc.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Tiled, depth);
    pipelined(walk, &mut ring, |slot, region| {
        let mut acc_r = acc.at(region);
        slot.consume(|a_s, b_s| {
            acc_r.mma(a_s, b_s, Semiring::SUM_PROD);
        });
    });
    acc.drain_cast_into(&mut c);
}

/// [`cmma_matmul_k_walk`] through the manual-mma instruction, whose fragment transports are
/// `io`'s.
#[cube(launch)]
fn mma_matmul_k_walk<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] io: MmaIOConfig,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.mma_accumulator::<E, E>(&a, io, Monoid::Sum);
    acc.zero();
    let walk = Walk::over(acc.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Strided, 1usize);
    pipelined(walk, &mut ring, |slot, region| {
        let mut acc_r = acc.at(region);
        slot.consume(|a_s, b_s| {
            acc_r.mma(a_s, b_s, Semiring::SUM_PROD);
        });
    });
    acc.drain_cast_into(&mut c);
}

/// [`mma_matmul_k_walk`] with a quantized lhs kept in its stored form by the stage, the manual
/// fragment load decoding each element as it reads.
#[cube(launch)]
fn mma_matmul_k_walk_quant<I: Numeric, E: Numeric>(
    a: &QuantTileArg<'_, I, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] io: MmaIOConfig,
    #[define(I)] _idtype: ElemType,
    #[define(E)] _edtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.mma_accumulator::<E, E>(&a, io, Monoid::Sum);
    acc.zero();
    let walk = Walk::over(acc.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Strided, 1usize);
    pipelined(walk, &mut ring, |slot, region| {
        let mut acc_r = acc.at(region);
        slot.consume(|a_s, b_s| {
            acc_r.mma(a_s, b_s, Semiring::SUM_PROD);
        });
    });
    acc.drain_cast_into(&mut c);
}

/// The multi-plane cmma stage: the outer K walk fills a shared stage cooperatively (`depth` in
/// flight), the inner level hands each plane its own fragment of the stage, resident across
/// every K step.
#[cube(launch)]
fn cmma_matmul_two_levels_planes<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.cmma_accumulator::<E, E>(&a, Monoid::Sum);
    acc.zero();
    let walk = Walk::over(acc.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Tiled, depth);
    pipelined(walk, &mut ring, |slot, region| {
        let acc_o = acc.at(region);
        slot.consume(|a_s, b_s| {
            for region in Walk::over(acc_o.op_space(a_s, b_s)) {
                let mut acc_p = acc_o.at(&region);
                acc_p.mma(&a_s.at(&region), &b_s.at(&region), Semiring::SUM_PROD);
            }
        });
    });
    acc.drain_cast_into(&mut c);
}

/// The multi-fragment partition: each plane owns a grid of fragments, resident across the outer
/// K walk; the innermost level selects one per region, reloading the operand fragments per
/// execute out of the stage.
#[cube(launch)]
fn cmma_matmul_three_levels_planes_fragments<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.cmma_accumulator::<E, E>(&a, Monoid::Sum);
    acc.zero();
    let walk = Walk::over(acc.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Tiled, depth);
    pipelined(walk, &mut ring, |slot, region| {
        let acc_o = acc.at(region);
        slot.consume(|a_s, b_s| {
            for region in Walk::over(acc_o.op_space(a_s, b_s)) {
                let acc_p = acc_o.at(&region);
                let a_p = a_s.at(&region);
                let b_p = b_s.at(&region);
                for frag in Walk::over(acc_p.op_space(&a_p, &b_p)).unrolled() {
                    let mut acc_f = acc_p.at(&frag);
                    acc_f.mma(&a_p.at(&frag), &b_p.at(&frag), Semiring::SUM_PROD);
                }
            }
        });
    });
    acc.drain_cast_into(&mut c);
}

/// The legacy register budget as a level structure: the K stage walk (staged, `depth` in
/// flight), the plane split, a contraction-step walk that only windows, an N walk loading one B
/// fragment per step beside the A column loaded once above it, and an M-only fragment walk
/// below. The two fragment walks select out of the plane's partition, so they unroll.
#[cube(launch)]
fn cmma_matmul_five_levels<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] depth: usize,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.cmma_accumulator::<E, E>(&a, Monoid::Sum);
    acc.zero();
    let walk = Walk::over(acc.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Tiled, depth);
    pipelined(walk, &mut ring, |slot, region| {
        let acc_o = acc.at(region);
        slot.consume(|a_s, b_s| {
            for region in Walk::over(acc_o.op_space(a_s, b_s)) {
                let acc_p = acc_o.at(&region);
                let a_p = a_s.at(&region);
                let b_p = b_s.at(&region);
                for step in Walk::over(acc_p.op_space(&a_p, &b_p)) {
                    let acc_k = acc_p.at(&step);
                    let a_k = a_p.at(&step);
                    let b_k = b_p.at(&step);
                    for col in Walk::over(acc_k.op_space(&a_k, &b_k)).unrolled() {
                        let acc_n = acc_k.at(&col);
                        let a_n = PlanePartition::cmma_fragments(&a_k.at(&col), &acc_n);
                        let b_n = PlanePartition::cmma_fragments(&b_k.at(&col), &acc_n);
                        for row in Walk::over(acc_n.op_space(&a_n, &b_n)).unrolled() {
                            let mut acc_m = acc_n.at(&row);
                            acc_m.mma(&a_n.at(&row), &b_n.at(&row), Semiring::SUM_PROD);
                        }
                    }
                }
            }
        });
    });
    acc.drain_cast_into(&mut c);
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
    #[comptime] space: Space,
    #[comptime] config: RegisterBlock,
    #[define(I)] _a_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    let walk = Walk::over(c.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Strided, 1usize);
    pipelined(walk, &mut ring, |slot, region| {
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
    #[comptime] space: Space,
    #[comptime] config: RegisterBlock,
    #[define(I)] _a_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    for region in Walk::over(c.op_space(&a, &b)) {
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
    #[comptime] space: Space,
    #[comptime] config: RegisterBlock,
    #[define(I)] _b_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile::<E>(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    let walk = Walk::over(c.op_space(&a, &b));
    let mut ring = Ring::smem(&walk, &a, &b, StageStorage::Strided, 1usize);
    pipelined(walk, &mut ring, |slot, region| {
        let mut c_r = c.at(region);
        slot.consume(|a_s, b_s| {
            c_r.mma_with(a_s, b_s, config, Semiring::SUM_PROD);
        });
    });
}

/// [`matmul_quant_lhs_in_place`]'s mirror: the quantized rhs served straight from global memory.
#[cube(launch)]
fn matmul_quant_rhs_in_place<I: Numeric, E: Numeric, V: Size>(
    a: &TileArg<'_, E, Const<1>>,
    b: &QuantTileArg<'_, I, Const<1>>,
    c: &TileArg<'_, E, V>,
    #[comptime] space: Space,
    #[comptime] config: RegisterBlock,
    #[define(I)] _b_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile::<E>(comptime!(space.clone()));
    let mut c = c.tile(space);
    c.zero();
    for region in Walk::over(c.op_space(&a, &b)) {
        let mut c_r = c.at(&region);
        c_r.mma_with(&a.at(&region), &b.at(&region), config, Semiring::SUM_PROD);
    }
}

/// [`promoted_matmul_in_place`] with a quantized lhs: a packed lhs contracting into a register
/// block, decoded per read.
#[cube(launch)]
fn promoted_matmul_quant_lhs_in_place<I: Numeric, E: Numeric, EA: Numeric>(
    a: &QuantTileArg<'_, I, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[comptime] config: RegisterBlock,
    #[define(I)] _a_dtype: ElemType,
    #[define(E)] _e_dtype: ElemType,
    #[define(EA)] _acc_dtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);
    let mut acc = c.block_accumulator::<EA, E>(&a, config, Monoid::Sum);
    acc.zero();
    for region in Walk::over(acc.op_space(&a, &b)) {
        let mut acc_r = acc.at(&region);
        acc_r.mma(&a.at(&region), &b.at(&region), Semiring::SUM_PROD);
    }
    acc.drain_cast_into(&mut c);
}

// ---- cmma fragment transit, by hand -------------------------------------------------

/// gmem → smem → cmma accumulator → smem → gmem: pure transit, no arithmetic.
#[cube(launch)]
fn cmma_roundtrip<E: Numeric>(
    input: &TileArg<'_, E, Const<1>>,
    output: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = input.tile(space);
    let space = comptime!(a.space.clone());

    let mut a_smem = MemData::smem(
        comptime!(space.clone()),
        1usize,
        StageStorage::Strided,
        0usize,
    );
    a_smem.copy_from(&a);
    sync_cube();

    let mut frag = CmmaData::<E>::fragment(
        MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(space.clone()),
    );
    frag.copy_from(&a_smem);

    let mut c_smem = MemData::smem(
        comptime!(space.clone()),
        1usize,
        StageStorage::Strided,
        0usize,
    );
    c_smem.copy_from(&frag);
    sync_cube();

    let mut c = output.tile(space);
    c.copy_from(&c_smem);
}

/// gmem A,B → smem → cmma A/B fragments; accumulator init from (zeroed) `c`, then
/// `cmma::execute` (`acc = A·B`), stored back through smem to gmem.
#[cube(launch)]
fn cmma_matmul<E: Numeric>(
    a: &TileArg<'_, E, Const<1>>,
    b: &TileArg<'_, E, Const<1>>,
    c: &TileArg<'_, E, Const<1>>,
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);

    let mut a_smem_tile = MemData::smem(
        comptime!(a.space.clone()),
        1usize,
        StageStorage::Strided,
        0usize,
    );
    a_smem_tile.copy_from(&a);

    let mut b_smem_tile = MemData::smem(
        comptime!(b.space.clone()),
        1usize,
        StageStorage::Strided,
        0usize,
    );
    b_smem_tile.copy_from(&b);

    let mut c_smem_tile = MemData::smem(
        comptime!(c.space.clone()),
        1usize,
        StageStorage::Strided,
        0usize,
    );
    c_smem_tile.copy_from(&c);
    sync_cube();

    let mut a_frag = CmmaData::<E>::fragment(
        MatrixIdent::A,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(a.space.clone()),
    );
    a_frag.copy_from(&a_smem_tile);

    let mut b_frag = CmmaData::<E>::fragment(
        MatrixIdent::B,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(b.space.clone()),
    );
    b_frag.copy_from(&b_smem_tile);

    let mut acc = CmmaData::<E>::fragment(
        MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(c.space.clone()),
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
    #[comptime] space: Space,
    #[define(E)] _dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);

    let mut a_smem = MemData::smem(
        comptime!(a.space.clone()),
        1usize,
        StageStorage::Strided,
        0usize,
    );
    a_smem.copy_from(&a);

    let mut b_smem = MemData::smem(
        comptime!(b.space.clone()),
        1usize,
        StageStorage::Strided,
        0usize,
    );
    b_smem.copy_from(&b);
    sync_cube();

    let mut a_frag = CmmaData::<E>::fragment(
        MatrixIdent::A,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(a.space.clone()),
    );
    a_frag.copy_from(&a_smem);

    let mut b_frag = CmmaData::<E>::fragment(
        MatrixIdent::B,
        8usize,
        8usize,
        8usize,
        MatrixLayout::ColMajor,
        comptime!(b.space.clone()),
    );
    b_frag.copy_from(&b_smem);

    let mut acc = CmmaData::<E>::fragment(
        MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(c.space.clone()),
    );
    acc.zero();

    acc.mma(&a_frag, &b_frag, Semiring::SUM_PROD);

    let mut c_smem = MemData::smem(
        comptime!(c.space.clone()),
        1usize,
        StageStorage::Strided,
        0usize,
    );
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
    #[comptime] space: Space,
    #[define(I)] _idtype: ElemType,
    #[define(E)] _edtype: ElemType,
) {
    let a = a.tile::<E>(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);

    let mut a_smem = MemData::smem(
        comptime!(a.space.clone()),
        1usize,
        StageStorage::Strided,
        0usize,
    );
    a_smem.copy_from(&a);

    let mut b_smem = MemData::smem(
        comptime!(b.space.clone()),
        1usize,
        StageStorage::Strided,
        0usize,
    );
    b_smem.copy_from(&b);

    let mut c_smem = MemData::smem(
        comptime!(c.space.clone()),
        1usize,
        StageStorage::Strided,
        0usize,
    );
    c_smem.copy_from(&c);
    sync_cube();

    let mut a_frag = CmmaData::<E>::fragment(
        MatrixIdent::A,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(a.space.clone()),
    );
    a_frag.copy_from(&a_smem);

    let mut b_frag = CmmaData::<E>::fragment(
        MatrixIdent::B,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(b.space.clone()),
    );
    b_frag.copy_from(&b_smem);

    let mut acc = CmmaData::<E>::fragment(
        MatrixIdent::Accumulator,
        8usize,
        8usize,
        8usize,
        MatrixLayout::RowMajor,
        comptime!(c.space.clone()),
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
    check_matmul(8, 8, 8, sequential(&[(M, 4), (N, 4), (K, 4)]), 1);
}

#[test]
fn matmul_one_tile_per_cube() {
    check_matmul(
        8,
        8,
        8,
        Partitioner::over(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (
                    M,
                    Distribution::Spatial {
                        scope: ComputeScope::Cube(CubeAxis::X),
                        spread: Spread::Contiguous,
                        coverage: Coverage::Instances(2),
                    },
                ),
                (
                    N,
                    Distribution::Spatial {
                        scope: ComputeScope::Cube(CubeAxis::Y),
                        spread: Spread::Contiguous,
                        coverage: Coverage::Instances(2),
                    },
                ),
                (K, Distribution::Sequential),
            ]),
        )
        .level(),
        1,
    );
}

/// The kernel owns the init: `c` comes out as `a·b`, whatever it held going in.
///
/// The whole contraction lands at the leaf here, so a single region writes each cell; the poison
/// the harness filled `c` with must be gone all the same.
#[test]
fn matmul_whole_k_at_the_leaf() {
    check_matmul(8, 8, 4, sequential(&[(M, 4), (N, 4), (K, 4)]), 1);
}

#[test]
fn matmul_reversed_walk_single_cube() {
    let client = cubecl::test_device().client();
    let (m, n, k, tile_edge) = (8usize, 8usize, 8usize, 4usize);
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(sequential(&[
        (M, 4),
        (N, 4),
        (K, 4),
    ]));
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .uniform(7, -100.0, 100.0);
    matmul_smem_ring_reversed::launch(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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
        Partitioner::over(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (
                    M,
                    Distribution::Spatial {
                        scope: ComputeScope::Cube(CubeAxis::X),
                        spread: Spread::Contiguous,
                        coverage: Coverage::TilesEach(2),
                    },
                ),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        )
        .level(),
        1,
    );
}

#[test]
fn matmul_interleaved_m_across_cubes() {
    check_matmul(
        16,
        8,
        8,
        Partitioner::over(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (
                    M,
                    Distribution::Spatial {
                        scope: ComputeScope::Cube(CubeAxis::X),
                        spread: Spread::Interleaved,
                        coverage: Coverage::Instances(2),
                    },
                ),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        )
        .level(),
        1,
    );
}

#[test]
fn matmul_double_buffered() {
    check_matmul(8, 8, 8, sequential(&[(M, 4), (N, 4), (K, 4)]), 2);
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

/// Drives [`matmul_smem_ring`] for `C = A @ B`: one level, both inputs staged, `depth` regions
/// in flight.
fn check_matmul(m: usize, n: usize, k: usize, partitioner: Partitioner, depth: usize) {
    let client = cubecl::test_device().client();
    let tile_edge = partitioner.edge(M);
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    // Poisoned, not zeroed: the kernel owns the init, so anything `c` held must be gone from the
    // result.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .uniform(7, -100.0, 100.0);

    matmul_smem_ring::launch(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(sequential(&[
        (M, tile_edge),
        (N, tile_edge),
        (K, k),
    ]));

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();

    matmul_smem_ring_accumulate::launch(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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
    let space = Space::new(&[(B, b), (M, m), (N, n), (K, k)]).with_partitioner(sequential(&[
        (B, batch_edge),
        (M, tile_edge),
        (N, tile_edge),
        (K, tile_edge),
    ]));
    let a = TileInput::builder(&client, space.project(&[B, M, K]))
        .tile(&[batch_edge, tile_edge, tile_edge])
        .arange();
    let rhs = TileInput::builder(&client, space.project(&[B, K, N]))
        .tile(&[batch_edge, tile_edge, tile_edge])
        .arange();
    let c = TileInput::builder(&client, space.project(&[B, M, N]))
        .tile(&[batch_edge, tile_edge, tile_edge])
        .zeros();

    matmul_smem_ring::launch(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        1,
        a.arg(),
        rhs.arg(),
        c.arg(),
        space,
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
        &[sequential(&[(B0, 1), (B1, 1), (M, 4), (N, 4), (K, 4)])],
    );
}

#[test]
fn matmul_broadcast_lhs_only() {
    // rhs broadcasts nothing (b0 = 1 makes B0 degenerate); lhs still broadcasts B1.
    check_matmul_broadcast(
        1,
        5,
        4,
        &[sequential(&[(B0, 1), (B1, 1), (M, 4), (N, 4), (K, 4)])],
    );
}

/// Both batch axes ride cube-Z at once: `B0` and `B1` are `Spatial { Cube(Z) }`, so
/// the launch puts their *product* on Z and the walk decodes one cube's `CUBE_POS_Z`
/// back into `(b0, b1)`. The same broadcast result as the sequential variants: this
/// is what lets CpuGemm parallelise the whole batch on Z.
#[test]
fn matmul_broadcast_two_batch_axes_on_z() {
    let z = || Distribution::Spatial {
        scope: ComputeScope::Cube(CubeAxis::Z),
        spread: Spread::Contiguous,
        coverage: Coverage::TilesEach(1),
    };
    check_matmul_broadcast(
        4,
        3,
        4,
        &[Partitioner::over(
            ByAxis::new(&[(B0, 1), (B1, 1), (M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (B0, z()),
                (B1, z()),
                (M, Distribution::Sequential),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        )
        .level()],
    );
}

/// The two-axis broadcast tiled across *two* levels: L0 walks the batch
/// (`batch_edge = 1`) and stages the whole `4×4` matrix, then L1 tiles that matrix
/// into `2×2` final tiles. The broadcast (omitted) batch axes must stay correct
/// through both `divide`s. The result is the same broadcast matmul.
#[test]
fn matmul_broadcast_multilevel() {
    check_matmul_broadcast(
        4,
        3,
        4,
        &[
            sequential(&[(B0, 1), (B1, 1), (M, 4), (N, 4), (K, 4)]),
            sequential(&[(B0, 1), (B1, 1), (M, 2), (N, 2), (K, 2)]),
        ],
    );
}

/// `C = A @ B` where the batch is two independent axes `B0`, `B1` and each operand
/// carries only one: `lhs ∈ {B0, M, K}`, `rhs ∈ {B1, K, N}`, `out ∈ {B0, B1, M, N}`.
/// Each operand omits the batch axis it broadcasts, and the kernel's `Space::merge`
/// fills the omitted axis back wholesale. Single tile per matrix (`t³`) with
/// `batch_edge = 1`, so each output batch element is its own walk point. Every level
/// stages, whatever the caller stacked.
fn check_matmul_broadcast(b0: usize, b1: usize, t: usize, partitioners: &[Partitioner]) {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    let space = partitioners.iter().fold(
        Space::new(&[(B0, b0), (B1, b1), (M, t), (N, t), (K, t)]),
        |s, p| s.with_partitioner(p.clone()),
    );
    let out = space.project(&[B0, B1, M, N]);
    let lhs = TileInput::builder(&client, space.project(&[B0, M, K]))
        .tile(&[1, t, t])
        .arange();
    let rhs = TileInput::builder(&client, space.project(&[B1, K, N]))
        .tile(&[1, t, t])
        .arange();
    let acc = TileInput::builder(&client, out.clone())
        .tile(&[1, 1, t, t])
        .zeros();

    let cube_count = out.cube_count();
    let cube_dim = CubeDim::new_single();
    match partitioners.len() {
        1 => matmul_smem_ring::launch(
            &client,
            cube_count,
            cube_dim,
            1,
            lhs.arg(),
            rhs.arg(),
            acc.arg(),
            space,
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
            space,
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
    check_matmul_cpu(8, 8, 8, sequential(&[(M, 4), (N, 4), (K, 4)]));
}

#[test]
fn matmul_cpu_big_k() {
    check_matmul_cpu(8, 8, 16, sequential(&[(M, 4), (N, 4), (K, 4)]));
}

#[test]
fn matmul_cpu_cores_split_m() {
    check_matmul_cpu(
        16,
        8,
        8,
        Partitioner::over(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (
                    M,
                    Distribution::Spatial {
                        scope: ComputeScope::Cube(CubeAxis::X),
                        spread: Spread::Contiguous,
                        coverage: Coverage::TilesEach(2),
                    },
                ),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        )
        .level(),
    );
}

#[test]
fn matmul_cpu_cores_split_m_planes() {
    check_matmul_cpu(
        16,
        8,
        8,
        Partitioner::over(
            ByAxis::new(&[(M, 4), (N, 4), (K, 4)]),
            ByAxis::new(&[
                (
                    M,
                    Distribution::Spatial {
                        scope: ComputeScope::Plane,
                        spread: Spread::Contiguous,
                        coverage: Coverage::TilesEach(2),
                    },
                ),
                (N, Distribution::Sequential),
                (K, Distribution::Sequential),
            ]),
        )
        .level(),
    );
}

/// The register leaf reads both operands where they lie: nothing is materialized and the walk is
/// the plain loop.
fn check_matmul_cpu(m: usize, n: usize, k: usize, partitioner: Partitioner) {
    let client = cubecl::test_device().client();
    let tile_edge = partitioner.edge(M);
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(partitioner);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    // Poisoned, not zeroed: the kernel owns `out = A·B` whatever the buffer held.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .uniform(4242, 10., 100.);

    matmul_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        REGISTER_BLOCK,
        Semiring::SUM_PROD,
        f32::elem_type_native(),
    );
    assert_tiled_matmul(&client, c.handle(), m, n, k, tile_edge);
}

/// The "global matmul" shape: M and N stay comptime (`Static`), only K is `Dynamic`, so its tile
/// count is resolved from the tensor at runtime while M/N fold and unroll. Exercises the mixed
/// `Static`/`Dynamic` path through `merged_space`/`Extents` that every `all_dynamic` caller skips.
/// Geometry and allocation use the concrete space; the kernel keys on the K-dynamic one.
#[test]
fn matmul_cpu_dynamic_k() {
    let client = cubecl::test_device().client();
    let (m, n, k, edge) = (8usize, 8usize, 16usize, 4usize);
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(sequential(&[
        (M, edge),
        (N, edge),
        (K, edge),
    ]));

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[edge, edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[edge, edge])
        .arange();
    // Poisoned, not zeroed: the kernel owns `out = A·B` whatever the buffer held.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[edge, edge])
        .uniform(4242, 10., 100.);

    matmul_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space.with_dynamic(&[K]),
        REGISTER_BLOCK,
        Semiring::SUM_PROD,
        f32::elem_type_native(),
    );
    assert_tiled_matmul(&client, c.handle(), m, n, k, edge);
}

/// N spread across a plane's lanes (`ComputeScope::Unit`): each lane owns a disjoint
/// column of the register-leaf output and contracts the whole K in registers: the
/// gemv-perpendicular mapping. A bare `lanes()` declares the split without the lane count;
/// [`Space::resolve_lanes`] (the launch's stamping pass) fills it from the hardware
/// `plane_size`, so the Unit axis rides the warp's lanes on the cube's X dim.
/// `plane_size == 1` on CPU degenerates to one lane doing all of N (still correct); the
/// win is on GPU where the warp's lanes divide N.
#[test]
fn register_matmul_unit_spread_n() {
    let client = cubecl::test_device().client();
    let plane_size = client.properties().hardware.plane_size_max as usize;

    let (m, k, nr) = (4usize, 8usize, 2usize);
    let n = plane_size * nr;
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        .level(|l| {
            l.distribute(lanes(plane_size), &[(N, nr)])
                .walk(&[(M, m), (K, k)]);
        })
        .build();

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    matmul_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        REGISTER_BLOCK,
        Semiring::SUM_PROD,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

// ---- padded stages ----------------------------------------------------------------

/// A scalar `K×N` source whose `N = 3` rows cannot be vectorized globally is padded into
/// four-wide shared-memory lines. The register gather scatters those lanes back into the scalar
/// `M×N` sink; the fourth lane is padding and must not consume the next row's first live value.
#[test]
fn matmul_padded_rhs_stage_into_scalar_sink() {
    check_padded_rhs_stage((2, 3, 2), vec![3.0, 4.0, 5.0, 9.0, 14.0, 19.0]);
}

/// The single-row shape, where a block column overhanging `N` has nowhere legal to land: with no
/// row after it, `block::commit`'s masked lanes are all that keeps the write inside the output.
#[test]
fn matmul_padded_rhs_stage_single_row_sink() {
    check_padded_rhs_stage((1, 3, 2), vec![3.0, 4.0, 5.0]);
}

/// A scalar `K×N` source with `N = 5` spanning two 4-wide shared-memory lines (total 8 lanes, 3
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
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        .level(|l| {
            l.walk(&[(M, m), (N, n), (K, k)]);
        })
        .build();
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();
    let launcher = space.launcher(&client);
    let a_op = launcher.arg(a.handle().binding()).subspace(&[M, K]).build();
    let b_op = launcher.arg(b.handle().binding()).subspace(&[K, N]).build();
    let c_op = launcher.arg(c.handle().binding()).subspace(&[M, N]).build();

    matmul_padded_rhs_stage::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a_op.arg(),
        b_op.arg(),
        c_op.arg(),
        launcher.space().clone(),
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
/// and sink keep this on the direct 2-D contraction path; its partial final lhs line must
/// contribute exactly three K values.
#[test]
fn matmul_padded_lhs_stage_direct_tail() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (2usize, 2usize, 3usize);
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        .level(|l| {
            l.walk(&[(M, m), (N, n), (K, k)]);
        })
        .level(|l| {
            l.walk(&[(M, m), (N, n), (K, k)]);
        })
        .build();
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();
    let launcher = space.launcher(&client);
    let a_op = launcher.arg(a.handle().binding()).subspace(&[M, K]).build();
    let b_op = launcher.arg(b.handle().binding()).subspace(&[K, N]).build();
    let c_op = launcher.arg(c.handle().binding()).subspace(&[M, N]).build();

    matmul_padded_lhs_stage_two_levels::launch(
        &client,
        launcher.cube_count(),
        launcher.cube_dim(),
        a_op.arg(),
        b_op.arg(),
        c_op.arg(),
        launcher.space().clone(),
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
    let space = Space::new(&[(M, m), (N, n), (K, k)])
        .with_partitioner(sequential(&[(M, 4), (N, 4), (K, 4)]))
        .with_partitioner(sequential(&[(M, 2), (N, 2), (K, 2)]));
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[final_edge, final_edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[final_edge, final_edge])
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[final_edge, final_edge])
        .zeros();
    matmul_two_levels_smem_then_in_place_reversed::launch(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        StageStorage::Strided,
        1,
        f32::elem_type_native(),
    );
    assert_tiled_matmul(&client, c.handle(), m, n, k, final_edge);
}

#[test]
fn matmul_multilevel_staged_then_staged() {
    check_matmul_multilevel(8, 8, 8, StageStorage::Strided, Inner::Staged(1), 1);
}

/// Double buffering at the higher level.
#[test]
fn matmul_multilevel_double_then_direct() {
    check_matmul_multilevel(8, 8, 8, StageStorage::Strided, Inner::Direct, 2);
}

/// Double buffering at the lower level.
#[test]
fn matmul_multilevel_staged_then_double() {
    check_matmul_multilevel(8, 8, 8, StageStorage::Strided, Inner::Staged(2), 1);
}

/// A storage-tiled stage on a register leaf: the stage layout knob off its default,
/// on any backend (each 4×4 stage cut into contiguous 2×2 blocks).
#[test]
fn matmul_multilevel_tiled_stage() {
    check_matmul_multilevel(8, 8, 8, StageStorage::Tiled, Inner::Direct, 1);
}

/// What the inner of two levels does with the outer stage: read its final tiles where they lie,
/// or restage them through a ring this deep.
#[derive(Clone, Copy)]
enum Inner {
    Direct,
    Staged(usize),
}

/// Drives the two-level kernels over `[4×4×4, 2×2×2]`: the outer level stages both operands laid
/// out as `storage`, `depth_outer` regions in flight; `inner` says what the second level does.
fn check_matmul_multilevel(
    m: usize,
    n: usize,
    k: usize,
    storage: StageStorage,
    inner: Inner,
    depth_outer: usize,
) {
    let client = cubecl::test_device().client();
    let final_edge = 2usize;
    let dtype = f32::elem_type_native();
    let space = Space::new(&[(M, m), (N, n), (K, k)])
        .with_partitioner(sequential(&[(M, 4), (N, 4), (K, 4)]))
        .with_partitioner(sequential(&[(M, 2), (N, 2), (K, 2)]));

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[final_edge, final_edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[final_edge, final_edge])
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[final_edge, final_edge])
        .zeros();

    match inner {
        Inner::Direct => matmul_two_levels_smem_then_in_place::launch(
            &client,
            space.cube_count(),
            CubeDim::new_single(),
            a.arg(),
            b.arg(),
            c.arg(),
            space,
            storage,
            depth_outer,
            dtype,
        ),
        Inner::Staged(depth_inner) => matmul_two_levels_smem_then_smem::launch(
            &client,
            space.cube_count(),
            CubeDim::new_single(),
            a.arg(),
            b.arg(),
            c.arg(),
            space,
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
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        .level(|l| {
            l.walk(&[(M, 4), (N, 4), (K, 4)]);
        })
        .level(|l| {
            l.walk(&[(M, 4), (N, 2), (K, 4)]);
        })
        .build();

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    matmul_two_levels_smem_then_smem::launch(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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
    let plain = Tiling::over(&[(M, m), (N, n), (K, k)])
        .level(|l| {
            l.walk(&[(M, 4), (N, 4), (K, 4)]);
        })
        .build();
    // The second level's edges are the first's: every axis's count is 1.
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        .level(|l| {
            l.walk(&[(M, 4), (N, 4), (K, 4)]);
        })
        .level(|l| {
            l.walk(&[(M, 4), (N, 4), (K, 4)]);
        })
        .build();
    assert_eq!(plain.partitioner().depth(), 1);
    assert_eq!(space.partitioner().depth(), 2);
    assert_ne!(space, plain);
    assert_eq!(space.cube_dim(&client), plain.cube_dim(&client));

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    matmul_two_levels_smem_then_in_place::launch(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        StageStorage::Strided,
        1,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

// A contraction cut at cube scope leaves each cube holding a slice of every output cell, and
// nothing combines them: a register-resident accumulator drains by storing, so the last cube to
// arrive erases the others, and one accumulating in place reads the cell, folds, and writes it
// back, which is a lost update between cubes. Both are refused (`SplitShare::validate`), and the
// refusal is unit-tested where it can be observed: `space::base` checks the share. Not here, for
// the reason `blocked.rs` gives: this one fires inside the kernel, on a worker thread, where
// `#[should_panic]` never sees it and the launch just returns zeros.

// ---- vectorized operands through the rings -------------------------------------------

/// Vectorized operands (2-wide lines) through the in-place path: gmem-only line-unit
/// addressing. Regression for the line-vs-scalar unit bug (worked on cubecl-cpu only).
#[test]
fn matmul_direct_vectorized() {
    let client = cubecl::test_device().client();
    let (m, n, k, edge) = (8usize, 8usize, 8usize, 4usize);
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(sequential(&[
        (M, edge),
        (N, edge),
        (K, edge),
    ]));
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();
    matmul_in_place::launch(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        2,
        2,
        2,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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

/// The same operands through a depth-2 ring: each region's fill overlaps the previous region's
/// compute. Depth is the only difference from the staged case above.
#[test]
fn matmul_double_buffered_vectorized() {
    check_matmul_vectorized((8, 8, 8), Staged::Both, 2);
}

/// Depth 3: two fills in flight over one compute. Regression for a ring whose drain leaves more
/// than one slot outstanding.
#[test]
fn matmul_triple_buffered_vectorized() {
    check_matmul_vectorized((8, 8, 8), Staged::Both, 3);
}

/// A depth deeper than the walk has regions: the prologue runs out of regions to prime and every
/// consume drains. Regression for the ring's `region < total` guards. Nine slots over eight
/// regions: two operands per slot, and Metal caps a kernel's threadgroup arguments at 31.
#[test]
fn matmul_buffered_deeper_than_the_walk() {
    check_matmul_vectorized((8, 8, 8), Staged::Both, 9);
}

/// A depth-2 ring whose walk cuts only `M`: `rhs` spans `K`/`N` alone, so the walk never moves its
/// window. It is filled once above the loop and its buffer serves both slots
/// (`WindowMode::Reused`), the only sound way for two slots to reuse one buffer, and why a stage
/// count is derived rather than stated.
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
/// staged one alone while the other is read where it lies, in every slot of the ring.
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
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(sequential(&[
        (M, tile_edge),
        (N, tile_edge),
        (K, tile_edge),
    ]));

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .tile(&[tile_edge, tile_edge])
        .arange();
    // Poisoned, not zeroed: the kernel owns the init.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .tile(&[tile_edge, tile_edge])
        .uniform(7, -100.0, 100.0);

    matmul_lhs_smem_ring::launch(
        &client,
        space.cube_count(),
        CubeDim::new_single(),
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        2,
        f32::elem_type_native(),
    );
    assert_tiled_matmul(&client, c.handle(), m, n, k, tile_edge);
}

/// Which operands a ring stages.
#[derive(Clone, Copy)]
enum Staged {
    Both,
    LhsOnly,
}

fn check_matmul_vectorized((m, n, k): (usize, usize, usize), staged: Staged, depth: usize) {
    let client = cubecl::test_device().client();
    let (edge, v) = (4usize, 2usize);
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(sequential(&[
        (M, edge),
        (N, edge),
        (K, edge),
    ]));

    let dtype = f32::elem_type_native();
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    match staged {
        Staged::Both => matmul_smem_ring::launch(
            &client,
            space.cube_count(),
            CubeDim::new_single(),
            v,
            a.arg(),
            b.arg(),
            c.arg(),
            space,
            depth,
            dtype,
        ),
        Staged::LhsOnly => matmul_lhs_smem_ring::launch(
            &client,
            space.cube_count(),
            CubeDim::new_single(),
            v,
            a.arg(),
            b.arg(),
            c.arg(),
            space,
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
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(sequential(&[
        (M, edge),
        (N, edge),
        (K, edge),
    ]));

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    // Poisoned: the kernel owns `out = A·B` whatever the buffer held.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let dtype = f32::elem_type_native();
    promoted_matmul_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(sequential(&[
        (M, edge),
        (N, edge),
        (K, edge),
    ]));

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .uniform(7, 1., 9.);
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .uniform(13, 1., 9.);
    // Poisoned: the kernel owns the init under this algebra too, and its identity is not zero.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    matmul_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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
/// at the lowest value, steps with `+`, and drains its lanes under the same fold.
#[test]
fn tropical_matmul_promoted() {
    let client = cubecl::test_device().client();
    let (m, n, k, edge) = (4usize, 4usize, 8usize, 4usize);
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(sequential(&[
        (M, edge),
        (N, edge),
        (K, edge),
    ]));

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .uniform(11, 1., 9.);
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .uniform(17, 1., 9.);
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let dtype = f32::elem_type_native();
    promoted_matmul_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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

/// The promoted register accumulator under the two-level cube/plane space a real gemm composes,
/// with **vectorized** operands (rhs and output in 2-wide lines). This is the case that once
/// failed to compile on the CPU backend, when the block was allocated scalar and re-viewed as
/// lines; the block is now allocated at its vector element (`Array<Vector<T, RA>>`), so the
/// store is a real vector write and the numbers are right on every runtime.
#[test]
fn register_matmul_promoted_cube_plane() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (4usize, 4usize, 16usize);
    let (leaf_m, leaf_n, leaf_k) = (2usize, 2usize, 4usize);
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        .level(|l| {
            l.distribute(cubes(CubeAxis::X), &[(M, m)])
                .distribute(cubes(CubeAxis::Y), &[(N, n)])
                .walk(&[(K, k)]);
        })
        .level(|l| {
            l.distribute(planes(), &[(M, leaf_m)])
                .distribute(planes(), &[(N, leaf_n)])
                .walk(&[(K, leaf_k)]);
        })
        .build();

    let dtype = f32::elem_type_native();
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    promoted_matmul_two_levels_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        // Rhs and output vectorized along N, as a real launch does: the tensor args stay
        // scalar-unit and the kernel's `Vector<E, V>` element carries the width.
        2,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        REGISTER_BLOCK,
        dtype,
        dtype,
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// A buffered level that *cuts* a promoted (fragment) accumulator: each region selects its own
/// block, so the ring's walk has to unroll and hand every region comptime coordinates.
///
/// The regression this guards is silent in both directions. `#[unroll(flag)]` only unrolls when
/// the macro sees `flag` as a comptime binding, and rolls the loop without complaint otherwise;
/// the lap arithmetic then has to fold, or the coordinates come out runtime even unrolled. Either
/// slip lands on `Tile::at`'s "must be walked with compile-time coordinates" panic. The other
/// unrolled shape, a fragment stage, needs a fragment leaf and so only runs on tensor-core
/// hardware ([`cmma_matmul_staged_n_walk_partition`]); this one runs everywhere.
#[test]
fn matmul_buffered_walk_cutting_a_fragment_accumulator_unrolls() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (4usize, 4usize, 8usize);
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        // L0: the whole output, K in two steps. The block mirrors this level's *sub-tile*, so
        // the accumulator's grid is only cut a level down.
        .level(|l| {
            l.walk(&[(M, 4), (N, 4), (K, 4)]);
        })
        // L1: the 2x2 cut of that partition, with both operands staged.
        .level(|l| {
            l.walk(&[(M, 2), (N, 2), (K, 2)]);
        })
        .build();

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel zeroes the promoted accumulator.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    block_matmul_two_levels_smem_below::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

// ---- lined lhs: the (line, lane) K walk --------------------------------------

/// A single-level space whose leaf takes the whole problem, the shape the lined-lhs and folded
/// tests drive.
fn lined_lhs_space(m: usize, n: usize, k: usize) -> Space {
    Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(sequential(&[(M, m), (N, n), (K, k)]))
}

/// The memory-backed leaf with the lhs lined 2-wide along `K`: two lanes per K-line, each
/// reaching its element by a comptime `extract` rather than a dynamic one.
#[test]
fn register_matmul_lined_lhs() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (4usize, 4usize, 8usize);
    let space = lined_lhs_space(m, n, k);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel owns `out = A·B` whatever the buffer held.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    matmul_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        2,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        REGISTER_BLOCK,
        Semiring::SUM_PROD,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// [`register_matmul_lined_lhs`] through the promoted block: same walk, same lanes, but the
/// accumulator never round-trips to the output between `K` steps.
#[test]
fn register_matmul_promoted_lined_lhs() {
    let client = cubecl::test_device().client();
    let (m, n, k) = (4usize, 4usize, 8usize);
    let space = lined_lhs_space(m, n, k);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    let dtype = f32::elem_type_native();
    promoted_matmul_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        // Lhs 2-wide along K, rhs and output 2-wide along N: both the lane fan-out and the
        // block's own line width are off their scalar case at once.
        2,
        2,
        2,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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
/// lanes are `K`-partials of one cell, and one horizontal fold collapses them. The rhs is declared
/// `[N, K]`, which is what puts its line on the contracted axis. `budget` sizes the block: too
/// small for the shape and the rolled body runs, indexing its local arrays at runtime.
fn check_folded_step(space: Space, (m, n, k): (usize, usize, usize), budget: usize) {
    let client = cubecl::test_device().client();
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[N, K]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel owns `out = A·Bᵀ` whatever the buffer held.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    matmul_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        4,
        4,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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

/// The N-D nest at a folded step: two contracted axes, both operands lined along the faster of
/// them. The reduce nest steps by the served width, so each step lands on a line start.
#[test]
fn register_matmul_folded_step_two_contracted_axes() {
    let client = cubecl::test_device().client();
    let (m, n, k1, k2) = (4usize, 4usize, 2usize, 4usize);
    let k = k1 * k2;
    let space = Tiling::over(&[(M, m), (N, n), (K, k1), (K2, k2)])
        .level(|l| {
            l.walk(&[(M, m), (N, n), (K, k1), (K2, k2)]);
        })
        .build();

    let a = TileInput::builder(&client, space.project(&[M, K, K2]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[N, K, K2]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    matmul_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        4,
        4,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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

    let space = lined_lhs_space(m, n, k);
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .arange();
    let b = TileInput::builder(&client, space.project(&[N, K]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    matmul_quant_lhs_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
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
        space,
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

// ---- the sub-plane fold: LaneShare::Group --------------------------------------

/// The space a rows-in-flight gemv cuts: the plane splits into aligned groups of `group_lanes`,
/// each group owning one output row and its lanes interleaving `K` between them. Every lane holds
/// a *partial* of its group's row, so the drain is a segmented reduction: `LaneShare::Group`,
/// where a whole-plane fold would be `LaneShare::Plane`.
///
/// `groups == 1` is the same space at `LaneShare::Plane`, which is the case already covered; the
/// point here is a plane carrying several cells at once.
fn lane_group_fold_space(plane_size: usize, group_lanes: usize, edge: usize, n: usize) -> Space {
    let groups = plane_size / group_lanes;
    Tiling::over(&[(M, groups), (N, n), (K, group_lanes * edge)])
        .level(|l| {
            l.distribute(lanes(groups), &[(M, 1)])
                .distribute(lanes(group_lanes).interleaved(), &[(K, edge)])
                .walk(&[(N, n)]);
        })
        .build()
}

/// The memory-backed leaf over the segmented fold, the control for
/// [`register_matmul_promoted_lane_group_fold`]. If this one fails the space itself is wrong and
/// the promoted result proves nothing.
#[test]
fn register_matmul_lane_group_fold() {
    let client = cubecl::test_device().client();
    let lanes = client.properties().hardware.plane_size_max as usize;
    let (group_lanes, edge, n) = (8usize, 4usize, 1usize);
    let (groups, k) = (lanes / group_lanes, group_lanes * edge);
    let m = groups;
    let space = lane_group_fold_space(lanes, group_lanes, edge, n);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    matmul_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        RegisterBlock::new(edge * n),
        Semiring::SUM_PROD,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// The same segmented fold through a **promoted** accumulator.
///
/// This is the case no other test covers: every promoted test in this file folds either nothing
/// (`Whole`) or the whole plane (`Plane`). A plane carrying one cell per group has to reduce
/// within each group and let each group's first lane write *its own row*, and the rows a group
/// owns are what the `M` cut hands it, which a block built before the walk descends has to be
/// told rather than assume.
#[test]
fn register_matmul_promoted_lane_group_fold() {
    let client = cubecl::test_device().client();
    let lanes = client.properties().hardware.plane_size_max as usize;
    let (group_lanes, edge, n) = (8usize, 4usize, 1usize);
    let (groups, k) = (lanes / group_lanes, group_lanes * edge);
    let (m, dtype) = (groups, f32::elem_type_native());
    let space = lane_group_fold_space(lanes, group_lanes, edge, n);

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    promoted_matmul_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        1,
        1,
        1,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        RegisterBlock::new(edge * n),
        Semiring::SUM_PROD,
        dtype,
        dtype,
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// A packed lhs contracts into a register block to the product its scales and values describe.
///
/// The decode belongs to the read, not to the leaf: `Tile::matrix_packed` dequantizes per read
/// for whichever leaf asks, so a promoted accumulator serves a quantized operand with nothing of
/// its own. What that is worth is only checkable against a reference the kernel had no hand in,
/// built on the host from the quantized values and their scales, since a leaf that decoded
/// wrongly and a reference that decoded the same way wrongly would agree.
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

    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(sequential(&[
        (M, edge),
        (N, edge),
        (K, edge),
    ]));

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .packed(&scheme, DequantAt::Read)
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    // Poisoned: the kernel owns `out = A·B` whatever the buffer held.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    promoted_matmul_quant_lhs_in_place::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
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
        space,
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

/// Round-trips a 16×16 tile through a tensor-core *accumulator* fragment with no
/// arithmetic: gmem → smem → cmma (load) → smem → gmem. Validates that the
/// `TileKind::Cmma` transit (`cmma::load_with_layout` / `cmma::store`) preserves data.
/// Tensor-core only: skipped on backends without cmma (wgpu/cpu); run with
/// `cargo test-metal`.
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
        space,
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
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    cmma_matmul::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    // `{N, K}`: the rhs transposed, so `b[j, p] = j·8 + p`.
    let b = TileInput::builder(&client, space.project(&[N, K]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    cmma_matmul_transposed_rhs::launch(
        &client,
        CubeCount::Static(1, 1, 1),
        CubeDim::new_3d(32, 1, 1),
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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
        space,
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
    check_cmma_matmul_k_walk(16, 1, 1, StageStorage::Tiled);
}

/// The double-buffered variant: four K regions rotating through two smem slots, the
/// accumulator fragment resident across all of them.
#[test]
fn cmma_matmul_double_buffered_k_walk() {
    check_cmma_matmul_k_walk(32, 2, 1, StageStorage::Tiled);
}

/// An odd region total (three K stages): the loop leaves the last region primed in slot 0;
/// the epilogue must publish and consume it.
#[test]
fn cmma_matmul_double_buffered_odd_k_walk() {
    check_cmma_matmul_k_walk(24, 2, 1, StageStorage::Tiled);
}

/// The K walk staged into a plain strided stage (the legacy `sync_full_strided` storage):
/// the cmma window transport reads through the layout stack either way.
#[test]
fn cmma_matmul_staged_k_walk_strided_stage() {
    check_cmma_matmul_k_walk(16, 1, 1, StageStorage::Strided);
}

/// The staged cmma K walk with operands served in 2-wide lines: the cooperative fill
/// moves lines, the cmma transport addresses the scalar buffer underneath.
#[test]
fn cmma_matmul_staged_k_walk_vectorized() {
    check_cmma_matmul_k_walk(16, 1, 2, StageStorage::Tiled);
}

/// The one level always stages, whatever its depth: a cmma leaf cannot consume the global inputs
/// directly, so the kernel first materializes them in shared memory.
fn check_cmma_matmul_k_walk(k: usize, depth: usize, v: usize, storage: StageStorage) {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, edge) = (8usize, 8usize, 8usize);
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        .level(|l| {
            l.walk(&[(M, edge), (N, edge), (K, edge)]);
        })
        .build();

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel zeroes the accumulator fragment.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    cmma_matmul_k_walk::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        v,
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        storage,
        depth,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// The manual/raw-mma instruction: the raw-mma twin of `cmma_matmul_staged_k_walk`: the same
/// open → zero → mma → drain kernel, but the contraction runs through `MmaDefinition::execute`
/// over register fragments rather than the cooperative `cmma::execute`. Gated on the backend
/// exposing the manual-mma feature (`features.matmul.mma`); uses the universal manual transport
/// (`MmaIOConfig::manual()`), so no `ldmatrix`/`stmatrix` path is taken. Run with `cargo
/// test-metal` / `test-cuda` on a backend that advertises manual mma.
#[test]
fn mma_matmul_8x8x8() {
    let client = cubecl::test_device().client();
    if !require_mma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, k, edge) = (8usize, 8usize, 8usize, 8usize);
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        .level(|l| {
            l.walk(&[(M, edge), (N, edge), (K, edge)]);
        })
        .build();

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel zeroes the accumulator fragment.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    mma_matmul_k_walk::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        MmaIOConfig::manual(),
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// The multi-plane cmma stage: a double-buffered K walk fills a shared `16×8`/`8×16`
/// stage cooperatively (cyclic across the cube's 128 units), and a plane-partitioned
/// inner level hands each of the 4 planes its own `8×8` fragment, resident across all
/// four K steps. Tensor-core only: run with `cargo test-metal`.
#[test]
fn cmma_matmul_plane_partitioned_stage() {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, k, edge) = (16usize, 16usize, 32usize, 8usize);
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        // L0: the whole `16×16` output per cube, K walked in `8`-deep stages, double-buffered.
        .level(|l| {
            l.walk(&[(M, m), (N, n), (K, edge)]);
        })
        // L1: the stage split one `8×8` fragment per plane.
        .level(|l| {
            l.distribute(planes(), &[(M, edge)])
                .distribute(planes(), &[(N, edge)])
                .walk(&[(K, edge)]);
        })
        .build();

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel zeroes the accumulator fragment.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    cmma_matmul_two_levels_planes::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        2,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// The multi-fragment partition: each of the 4 planes owns a 2×2 partition of 8³
/// fragments, resident across a double-buffered K walk; the fragment level reads the stage
/// where it lies, so the unrolled walk reloads operand fragments per execute (no restaging).
/// Tensor-core only; run with `cargo test-metal`.
#[test]
fn cmma_matmul_multi_fragment_partition() {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, k) = (32usize, 32usize, 32usize);
    let (part, i, stage_k) = (16usize, 8usize, 16usize);
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        // L0: whole output per cube, K walked in `stage_k`-deep double-buffered stages.
        .level(|l| {
            l.walk(&[(M, m), (N, n), (K, stage_k)]);
        })
        // L1: the stage split one `part×part` partition per plane (2×2 planes).
        .level(|l| {
            l.distribute(planes(), &[(M, part)])
                .distribute(planes(), &[(N, part)])
                .walk(&[(K, stage_k)]);
        })
        // L2: the partition level, 2×2 fragments per plane, 2 K sub-tiles.
        .level(|l| {
            l.walk(&[(M, i), (N, i), (K, i)]);
        })
        .build();

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel zeroes the accumulator fragments.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    cmma_matmul_three_levels_planes_fragments::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        a.arg(),
        b.arg(),
        c.arg(),
        space,
        2,
        f32::elem_type_native(),
    );
    assert_matmul_arange(&client, c.handle(), m, n, k);
}

/// The legacy register budget as a level structure: a contraction-step walk (windowing only),
/// an N-walk loading one B fragment per step while the A column loads once above it, and an
/// M-only fragment walk below. Exercises sub-block partition selection (the N-walk's regions
/// each own a column of the accumulator) and the unrolled fragment walks. Tensor-core only.
#[test]
fn cmma_matmul_staged_n_walk_partition() {
    let client = cubecl::test_device().client();
    if !require_cmma_8x8x8_f32(&client) {
        return;
    }

    let (m, n, k) = (32usize, 32usize, 32usize);
    let (part, i, stage_k) = (16usize, 8usize, 16usize);
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        // L0: whole output per cube, K walked in `stage_k`-deep double-buffered stages; the
        // accumulator opened here lives across the whole K walk below.
        .level(|l| {
            l.walk(&[(M, m), (N, n), (K, stage_k)]);
        })
        // L1: the stage split one `part×part` partition per plane (2×2 planes).
        .level(|l| {
            l.distribute(planes(), &[(M, part)])
                .distribute(planes(), &[(N, part)])
                .walk(&[(K, stage_k)]);
        })
        // L2: the contraction-step walk, windowing only.
        .level(|l| {
            l.walk(&[(M, part), (N, part), (K, i)]);
        })
        // L3: the N-walk: one B fragment per step, the A column loaded once above it.
        .level(|l| {
            l.walk(&[(M, part), (N, i), (K, i)]);
        })
        // L4: the M-only fragment walk.
        .level(|l| {
            l.walk(&[(M, i), (N, i), (K, i)]);
        })
        .build();

    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    // Poisoned, not zeroed: the kernel zeroes the accumulator fragments.
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .uniform(4242, 10., 100.);

    cmma_matmul_five_levels::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
        a.arg(),
        b.arg(),
        c.arg(),
        space,
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
/// a line's coordinate counts lines while its scale block is cut in elements: the widening
/// [`ScaleLayout`] does. Two lines per `bk = 4` block, so a stage's scale still changes mid-fill.
/// Tensor-core only.
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
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        .level(|l| {
            l.walk(&[(M, edge), (N, edge), (K, edge)]);
        })
        .build();

    let a_dtype = ElemType::from_quant_value(scheme.value);
    let (lo, hi) = scheme.value.range();
    let (a_input, a_host) = TestInput::builder(client.clone(), shape![m, k])
        .dtype(a_dtype)
        .uniform(0x1, lo, hi)
        .generate_with_f32_host_data();
    let scales = TestInput::builder(client.clone(), shape![scales_shape.0, scales_shape.1])
        .custom(scale_vals.clone())
        .generate_without_host_data();

    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    cmma_matmul_k_walk_quant::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
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
        space,
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
/// `i8`, and the fragment load decodes each element through the quant-transparent matrix view.
/// The cmma twin of this test has no choice but `DequantAt::Load`, because its fragment load
/// takes a raw window; the manual transport addresses one element at a time, so it can decode.
/// Same numbers, a stage that is a quarter the size.
#[test]
fn mma_matmul_quant_until_read() {
    let client = cubecl::test_device().client();
    // The shape, not just the feature; see `require_mma_8x8x8_f32`. The `f32` triple, not the
    // stored `i8` one, and `8x8x8`, not `8x8x16`: `K = 16` is the *walk*, walked 8 deep, and
    // `A` decodes at the read, so the instruction this leaf reaches for is the same f32 `8x8x8`
    // the plain manual-mma test runs.
    if !require_mma_8x8x8_f32(&client) || !require_native_i8(&client) {
        return;
    }

    let (m, n, k, edge) = (8usize, 8usize, 16usize, 8usize);
    let space = Tiling::over(&[(M, m), (N, n), (K, k)])
        .level(|l| {
            l.walk(&[(M, edge), (N, edge), (K, edge)]);
        })
        .build();

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

    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();

    mma_matmul_k_walk_quant::launch(
        &client,
        space.cube_count(),
        space.cube_dim(&client),
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
        space,
        MmaIOConfig::manual(),
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
// Every other quant matmul above runs on tensor cores and skips where cmma is absent, which
// is everywhere the memory-bound GEMV actually lives. These pin the other leaf: the staged
// kernel stages `A`'s *packed storage words* into smem, and the software instruction
// dequantizes each read out of smem: no f32 inflation of the stage, no promotion, no cmma, no
// i8 needed for the packed cases (the binding is a `u32`).

/// One level cutting `tm×tn×tk` register-leaf tiles: the shape `check_matmul` drives, minus the
/// storage tiling (operands stay plain strided).
fn register_partitioner(tm: usize, tn: usize, tk: usize) -> Partitioner {
    sequential(&[(M, tm), (N, tn), (K, tk)])
}

/// Native i8 `A`, one scale per `bm`-row block, through the register leaf.
#[test]
fn register_matmul_quant_native_block_m() {
    run_register_matmul_quant_native(Serve::Staged);
}

/// Native i8 `A` served DIRECTLY through the register leaf (Keystone K): nothing is staged, so
/// the leaf reads i8 straight from gmem and scales per read. The native + lhs-arm twin of the
/// packed-rhs [`register_matmul_quant_rhs_direct_serve_gemv`]; together they exercise every
/// branch of the leaf's quant dispatch (lhs/rhs × native/packed).
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
        register_partitioner(4, 4, 4),
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
        register_partitioner(4, 4, tk),
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
/// [`Serve::Direct`] serves it straight from gmem. Either way through the leaf's
/// `matrix_transparent`, with no dequantized f32 stage.
#[allow(clippy::too_many_arguments)]
fn run_register_matmul_quant(
    client: Client,
    (m, n, k): (usize, usize, usize),
    plan: Partitioner,
    serve: Serve,
    a_arg: TensorArg,
    a_dtype: ElemType,
    scheme: QuantScheme,
    scales_arg: TensorArg,
    scale_vals: Vec<f32>,
    bm: usize,
    q: Vec<f32>,
) {
    let space = Space::new(&[(M, m), (N, n), (K, k)]).with_partitioner(plan);

    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .arange();
    let c = TileInput::builder(&client, space.project(&[M, N]))
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
            space,
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
            space,
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
// The gemv production shape: the *weight* is the streamed RHS at `(K, N) = (d_in, d_out)`,
// packed along `d_out` (the innermost axis) with one scale per `(k, N-group)` block
// (`[1, bn]`). A stays float. The RHS's served width drives the accumulator's line width
// in the register instruction, so `C` is launched at the same width.

/// Packed-u32 Q8S `B` (4 values per word along `N`), scales `[1, bn]`: the exact scheme
/// family `metabolic`'s gemv ships (`q8s`, packed-u32, block scales along `d_out`).
#[test]
fn register_matmul_quant_rhs_packed_q8() {
    let client = cubecl::test_device().client();
    let space =
        Space::new(&[(M, 8), (N, 8), (K, 8)]).with_partitioner(register_partitioner(4, 4, 4));
    run_register_matmul_quant_rhs(
        client,
        space,
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
    let space =
        Space::new(&[(M, 8), (N, 16), (K, 8)]).with_partitioner(register_partitioner(4, 8, 4));
    run_register_matmul_quant_rhs(
        client,
        space,
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
    let space =
        Space::new(&[(M, 1), (N, 8), (K, 8)]).with_partitioner(register_partitioner(1, 4, 4));
    run_register_matmul_quant_rhs(
        client,
        space,
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
    let space = Tiling::over(&[(M, 1), (N, 16), (K, 8)])
        .level(|l| {
            l.distribute(cubes(CubeAxis::X), &[(N, 4)])
                .walk(&[(M, 1), (K, 4)]);
        })
        .build();
    run_register_matmul_quant_rhs(
        client,
        space,
        QuantValue::Q8S,
        4,
        DequantAt::Read,
        Serve::Staged,
        None,
    );
}

/// Direct-serve the quantized RHS weight (Keystone K): nothing is staged, so the register leaf
/// reads the packed weight straight from gmem and dequantizes *per read* through
/// [`matrix_transparent`]: the sync-free `m = 1` decode path. The `_rhs_*` tests above are all
/// staged: they stage the weight's *packed words* into smem (plus its scales) and dequantize per
/// read out of smem. Same answer; direct avoids even the smem round-trip.
#[test]
fn register_matmul_quant_rhs_direct_serve_gemv() {
    let client = cubecl::test_device().client();
    let space = Tiling::over(&[(M, 1), (N, 8), (K, 8)])
        .level(|l| {
            l.walk(&[(M, 1), (N, 4), (K, 4)]);
        })
        .build();
    run_register_matmul_quant_rhs(
        client,
        space,
        QuantValue::Q8S,
        4,
        DequantAt::Read,
        Serve::Direct,
        None,
    );
}

/// The Goal path: a staged packed weight whose smem stage holds the *packed u32 words*, not a
/// dequantized f32 stage. A four-region K-walk (`k = 16`, `tk = 4`) with block `[1, bn]` scales
/// (distinct along K), so each region refills both the staged packed words and the staged
/// scales, and the leaf dequantizes per read out of smem via [`matrix_transparent`]. This is the
/// batched weight-streaming case the change targets: the contrast to the f32-inflated stage the
/// cmma leaf still uses, and to the sync-free direct serve above.
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
/// stage holds served values, so it costs the served-to-stored ratio in shared memory and the
/// decode happens once per element rather than per read. The fork a register leaf may take and a
/// cmma leaf is forced into; same numbers either way, which is the point of checking it.
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
/// `stage_scales` writes `global * local` into the smem scale grid, so the reads below see
/// effective one-level scales. The expectation carries the global scale, so a fold that never
/// happens (or happens twice) fails by that factor.
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
fn four_region_k_walk() -> Space {
    Tiling::over(&[(M, 4), (N, 8), (K, 16)])
        .level(|l| {
            l.walk(&[(M, 4), (N, 4), (K, 4)]);
        })
        .build()
}

/// Drive the quantized-rhs register kernels and check
/// `C[i,j] = Σ_p A[i,p] · q_b[p,j] · scale[p, j/bn]`.
#[allow(clippy::too_many_arguments)]
fn run_register_matmul_quant_rhs(
    client: Client,
    space: Space,
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

    let (m, n, k) = (space.extent(M), space.extent(N), space.extent(K));
    let a = TileInput::builder(&client, space.project(&[M, K]))
        .untiled()
        .arange();
    // The weight and its per-(k, N-group) scales, minted together.
    let b = TileInput::builder(&client, space.project(&[K, N]))
        .untiled()
        .packed(&mint_scheme, dequant_at)
        .arange();
    let global_scale = global.map(|g| {
        TestInput::builder(client.clone(), shape![1])
            .custom(vec![g])
            .generate_without_host_data()
    });
    let c = TileInput::builder(&client, space.project(&[M, N]))
        .untiled()
        .zeros();
    let b_dtype = u32::elem_type_native();
    let e_dtype = f32::elem_type_native();

    // Routine-like: the launcher derives geometry and argument wiring from the space; the
    // quantized RHS goes through the source builder, which binds it at the storage width.
    let launcher = space.launcher(&client);
    let a_op = launcher.arg(a.handle().binding()).subspace(&[M, K]).build();
    let mut scales = vec![b.scales_binding()];
    scales.extend(global_scale.map(|g| g.binding()));
    let b_op = launcher
        .arg(b.tile.handle().binding())
        .subspace(&[K, N])
        .vectorize(pack)
        .quantized(&scales, scheme, dequant_at)
        .build();
    // The register instruction lines the accumulator at the RHS's served width.
    let c_op = launcher
        .arg(c.handle().binding())
        .subspace(&[M, N])
        .vectorize(pack)
        .build();
    match serve {
        Serve::Staged => matmul_quant_rhs_smem_ring::launch(
            &client,
            launcher.cube_count(),
            launcher.cube_dim(),
            c_op.vector_size,
            a_op.arg(),
            b_op.arg(),
            c_op.arg(),
            launcher.space().clone(),
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
            b_op.arg(),
            c_op.arg(),
            launcher.space().clone(),
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
