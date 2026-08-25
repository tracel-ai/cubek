//! Unit tests for [`Space`]

use cubecl::ir::{ElemType, FloatKind};
use cubecl::prelude::*;
use cubecl::quant::scheme::QuantScheme;
use cubek_tile::{
    Axis, Buffering, ByAxis, Cut, Distribution, Instruction, Operand, OperandSet, Partitioner,
    Residence, Space, Stage, Tiling, WalkOrder,
};

// Matmul-style axis labels reused across the cases below. `B0`/`B1` are two
// independent batch axes (a batch is just ordinary axes; broadcasting is omission).
const M: Axis = Axis(0);
const N: Axis = Axis(1);
const K: Axis = Axis(2);
const B0: Axis = Axis(3);
const B1: Axis = Axis(4);

// ---- Space ----------------------------------------------------------------

#[test]
fn new_builds_plain_axes() {
    let space = Space::new(&[(M, 4), (N, 8), (K, 2)]);
    assert_eq!(space.rank(), 3);
    assert_eq!(space.extent(M), 4);
    assert_eq!(space.extent(N), 8);
    assert_eq!(space.extent(K), 2);
}

#[test]
fn project_keeps_listed_axes_in_order() {
    let space = Space::new(&[(B0, 12), (M, 16), (K, 8)]);
    let lhs = space.project(&[B0, M, K]);
    assert_eq!(lhs.rank(), 3);
    assert_eq!(lhs.extent(B0), 12);
    assert_eq!(lhs.extent(M), 16);

    // An operand broadcasts a batch axis by simply leaving it out of the projection.
    let dropped = space.project(&[M, K]);
    assert_eq!(dropped.rank(), 2);
    assert!(!dropped.contains(B0));
}

#[test]
fn merge_assembles_two_batch_broadcast() {
    // lhs carries B0, rhs carries B1; each omits (broadcasts) the other's batch axis.
    // The merge rebuilds the full {B0, B1, M, N, K} operation space.
    let lhs = Space::new(&[(B0, 4), (M, 16), (K, 8)]);
    let rhs = Space::new(&[(B1, 3), (K, 8), (N, 4)]);
    let out = Space::new(&[(B0, 4), (B1, 3), (M, 16), (N, 4)]);

    let merged = Space::merge(&[&out, &lhs, &rhs]);
    assert_eq!(merged.rank(), 5);
    assert_eq!(merged.extent(B0), 4);
    assert_eq!(merged.extent(B1), 3);
    assert_eq!(merged.extent(K), 8);
    // First-appearance order: out's axes lead, K (only on the operands) comes last.
    assert_eq!(merged.axis_at(0), B0);
    assert_eq!(merged.axis_at(1), B1);
    assert_eq!(merged.axis_at(4), K);
}

#[test]
fn merge_shared_size_one_axis_broadcasts() {
    // A shared axis where one side is size 1 broadcasts to the other (numpy rule).
    let lhs = Space::new(&[(B0, 1), (M, 16)]);
    let rhs = Space::new(&[(B0, 4), (M, 16)]);
    assert_eq!(Space::merge(&[&lhs, &rhs]).extent(B0), 4);
}

#[test]
fn merge_omitted_axis_broadcasts_wholesale() {
    let lhs = Space::new(&[(B0, 12), (M, 16), (K, 8)]);
    let rhs = Space::new(&[(K, 8), (N, 4)]);

    let merged = Space::merge(&[&lhs, &rhs]);
    assert!(merged.contains(B0));
    assert_eq!(merged.extent(B0), 12);
}

#[test]
fn merge_plain_shared_axis_agrees() {
    let lhs = Space::new(&[(M, 16), (K, 8)]);
    let rhs = Space::new(&[(K, 8), (N, 4)]);
    let merged = Space::merge(&[&lhs, &rhs]);
    assert_eq!(merged.extent(K), 8);
    assert_eq!(merged.rank(), 3);
}

#[test]
#[should_panic(expected = "conflicting extents")]
fn merge_conflicting_extent_panics() {
    let lhs = Space::new(&[(M, 16), (K, 8)]);
    let rhs = Space::new(&[(K, 4), (N, 4)]); // K disagrees: 8 vs 4
    let _ = Space::merge(&[&lhs, &rhs]);
}

// ---- Space::divide (the tiling scheme) ------------------------------------

fn sequential(edges: &[(Axis, usize)]) -> Partitioner {
    let dists = edges
        .iter()
        .map(|&(a, _)| (a, Distribution::Sequential))
        .collect::<Vec<_>>();
    Partitioner::row_major(ByAxis::new(edges), ByAxis::new(&dists)).buffered(Buffering::SINGLE)
}

#[test]
fn divide_cuts_each_axis_to_its_sub_tile_edge() {
    let partitioner = sequential(&[(M, 4), (N, 3), (K, 2)]);
    let space = Space::new(&[(M, 16), (N, 12), (K, 8)]).with_partitioner(partitioner);

    let tile = space.divide();
    assert_eq!(tile.extent(M), 4);
    assert_eq!(tile.extent(N), 3);
    assert_eq!(tile.extent(K), 2);
    assert_eq!(tile.rank(), 3);
}

#[test]
fn divide_chains_into_a_multi_level_scheme() {
    // The scheme is a tree of spaces
    let space = Space::new(&[(M, 64), (N, 64)]).with_partitioner(sequential(&[(M, 16), (N, 16)]));
    let level1 = space.divide();
    let level2 = level1
        .clone()
        .with_partitioner(sequential(&[(M, 4), (N, 4)]))
        .divide();

    assert_eq!(level1.extent(M), 16);
    assert_eq!(level2.extent(M), 4);
    assert_eq!(level2.extent(N), 4);
}

// ---- Space::overhangs ------------------------------------------------------

/// A cpu_gemm-shaped two-level scheme: a cube tile of `planes × leaf` leaves over `(m, n, k)`,
/// K cut to its full extent at the cube level (sequential contraction) then to `leaf_k`.
fn cpu_gemm_space(m: usize, n: usize, k: usize) -> Space {
    let (leaf_m, leaf_n, leaf_k) = (8, 8, 4);
    let (planes_m, planes_n) = (2, 4);
    Space::new(&[(M, m), (N, n), (K, k)])
        .with_partitioner(sequential(&[
            (M, planes_m * leaf_m),
            (N, planes_n * leaf_n),
            (K, k),
        ]))
        .with_partitioner(sequential(&[(M, leaf_m), (N, leaf_n), (K, leaf_k)]))
}

#[test]
fn overhangs_matches_cpu_gemm_checks() {
    // Every level divides: cube tiles 16×32, leaves 8×8×4.
    let space = cpu_gemm_space(64, 64, 16);
    assert!(!space.overhangs(M));
    assert!(!space.overhangs(N));
    assert!(!space.overhangs(K));

    // m = 40 is not a multiple of the cube tile (16): M overhangs (cpu_gemm's check_m).
    // Within a cube the plane split is exact, so the leaf level adds nothing.
    assert!(cpu_gemm_space(40, 64, 16).overhangs(M));

    // K's cube-level cut is its full extent (always divides); k = 18 fails only at the
    // leaf (leaf_k = 4): the deeper level alone drives the overhang (cpu_gemm's check_k).
    let space = cpu_gemm_space(64, 64, 18);
    assert!(space.overhangs(K));
    assert!(!space.overhangs(M));
}

#[test]
fn overhangs_when_a_deeper_edge_misdivides_its_parent() {
    // Top divides (32 % 16 == 0) but the second edge doesn't divide the first (16 % 3 != 0):
    // the parent edge, not the top extent, is what each level must divide.
    let space = Space::new(&[(M, 32)])
        .with_partitioner(sequential(&[(M, 16)]))
        .with_partitioner(sequential(&[(M, 3)]));
    assert!(space.overhangs(M));
}

#[test]
fn overhangs_final_space_never() {
    // No partitioner level: nothing to misdivide.
    assert!(!Space::new(&[(M, 7)]).overhangs(M));
}

#[test]
#[should_panic(expected = "concrete space")]
fn overhangs_dynamic_axis_panics() {
    let space = Space::new(&[(M, 64)])
        .with_partitioner(sequential(&[(M, 16)]))
        .all_dynamic();
    let _ = space.overhangs(M);
}

#[test]
fn with_partitioner_stacks_levels_and_divide_descends() {
    // Stacking partitioners builds the whole multi-level scheme up front
    let space = Space::new(&[(M, 64), (N, 64)])
        .with_partitioner(sequential(&[(M, 16), (N, 16)]))
        .with_partitioner(sequential(&[(M, 4), (N, 4)]));
    assert!(!space.is_final());

    let level1 = space.divide(); // head (16×16) consumed, 4×4 remains
    assert_eq!(level1.extent(M), 16);
    assert!(!level1.is_final());

    let final_space = level1.divide(); // 4×4 consumed
    assert_eq!(final_space.extent(M), 4);
    assert_eq!(final_space.extent(N), 4);
    assert!(final_space.is_final());

    // `final_space()` shortcuts straight to the bottom of the stack.
    assert_eq!(space.final_space().extent(M), 4);
    assert!(space.final_space().is_final());
}

// ---- Tiling::over -----------------------------------------------------------

struct MatmulOperands {
    a: Operand,
    b: Operand,
    out: Operand,
}

impl OperandSet for MatmulOperands {
    fn each(&mut self) -> impl Iterator<Item = &mut Operand> {
        [&mut self.a, &mut self.b, &mut self.out].into_iter()
    }
}

fn matmul_operands() -> MatmulOperands {
    let f32t = f32::elem_type_native();
    MatmulOperands {
        a: Operand::new(&[M, K], f32t),
        b: Operand::new(&[K, N], f32t),
        out: Operand::new(&[M, N], f32t),
    }
}

/// The two builders cannot drift: an operand-threaded build partitions exactly as the plain
/// chain does, so migrating a caller changes nothing about its space.
#[test]
fn over_builds_the_space_plain_tiling_would() {
    let plain = Tiling::new()
        .extents(&[(M, 64), (N, 64), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(16))
                .axis(N, Cut::sequential(32))
                .axis(K, Cut::sequential(16))
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(8))
                .axis(N, Cut::sequential(8))
                .axis(K, Cut::sequential(4))
        })
        .build();

    let mut ops = matmul_operands();
    let space = Tiling::over(&mut ops, &[(M, 64), (N, 64), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::sequential(16))
                .axis(N, Cut::sequential(32))
                .axis(K, Cut::sequential(16));
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::sequential(8))
                .axis(N, Cut::sequential(8))
                .axis(K, Cut::sequential(4));
        })
        .build();

    assert_eq!(space.partitioner(), plain.partitioner());
    assert_eq!(space.rank(), plain.rank());
    assert_eq!(space.extent(M), plain.extent(M));
    assert_eq!(space.extent(N), plain.extent(N));
    assert_eq!(space.extent(K), plain.extent(K));
}

/// Omission is a statement: a level that says nothing leaves the operand in place at the type
/// it already holds, so every stages is total without a default anyone has to remember.
#[test]
fn over_seals_stages_and_omission_is_in_place() {
    let f32t = f32::elem_type_native();
    let mut ops = matmul_operands();
    let _ = Tiling::over(&mut ops, &[(M, 64), (N, 64), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, Cut::sequential(16))
                .axis(N, Cut::sequential(16))
                .axis(K, Cut::sequential(16));
            o.a.stage(Residence::Smem);
            o.b.stage(Residence::Smem);
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, Cut::sequential(8))
                .axis(N, Cut::sequential(8))
                .axis(K, Cut::sequential(4));
            o.out.stage(Residence::Register);
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::sequential(4))
                .axis(N, Cut::sequential(4))
                .axis(K, Cut::sequential(4));
        })
        .build();

    let stage = |residence, dtype| Stage { residence, dtype };
    assert_eq!(
        ops.a.stages(),
        &[
            stage(Residence::Smem, f32t),
            stage(Residence::InPlace, f32t),
            stage(Residence::InPlace, f32t),
        ]
    );
    assert_eq!(
        ops.out.stages(),
        &[
            stage(Residence::InPlace, f32t),
            stage(Residence::Register, f32t),
            stage(Residence::InPlace, f32t),
        ]
    );
}

/// The type column is the data flow: moves keep the operand's type, a conversion re-types it
/// exactly where stated, and the padding below a conversion carries the converted type, so
/// reading one operand's stages top to bottom is reading what its bytes are at every level.
#[test]
fn over_type_column_moves_then_converts() {
    let q4 = ElemType::Float(FloatKind::E2M1x2);
    let f32t = f32::elem_type_native();
    let ops = MatmulOperands {
        a: Operand::new(&[M, K], f32t),
        b: Operand::new(&[K, N], q4).quantized(QuantScheme::default()),
        out: Operand::new(&[M, N], f32t),
    };
    let mut ops = ops;
    let _ = Tiling::over(&mut ops, &[(M, 64), (N, 64), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, Cut::sequential(16))
                .axis(N, Cut::sequential(16))
                .axis(K, Cut::sequential(16));
            o.b.stage(Residence::Smem);
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, Cut::sequential(8))
                .axis(N, Cut::sequential(8))
                .axis(K, Cut::sequential(4));
            o.b.stage_as(Residence::Register, f32t);
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::sequential(4))
                .axis(N, Cut::sequential(4))
                .axis(K, Cut::sequential(4));
        })
        .build();

    let stage = |residence, dtype| Stage { residence, dtype };
    assert_eq!(
        ops.b.stages(),
        &[
            stage(Residence::Smem, q4),       // move: packed words
            stage(Residence::Register, f32t), // the conversion, right here
            stage(Residence::InPlace, f32t),
        ]
    );
    assert_eq!(ops.b.current_dtype(), f32t);
    assert!(ops.b.quant().is_some());
}

/// `stage_as` is total in the type: stating the type the operand already holds is the move
/// `stage` states, so a caller whose type is computed says it once and never branches.
#[test]
fn over_stage_as_same_type_is_the_move() {
    let f32t = f32::elem_type_native();
    let mut ops = matmul_operands();
    let _ = Tiling::over(&mut ops, &[(M, 64), (N, 64), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, Cut::sequential(16))
                .axis(N, Cut::sequential(16))
                .axis(K, Cut::sequential(16));
            o.a.stage_as(Residence::Smem, f32t);
            o.b.stage(Residence::Smem);
        })
        .build();

    assert_eq!(ops.a.stages(), ops.b.stages());
    assert_eq!(
        ops.a.stages(),
        &[Stage {
            residence: Residence::Smem,
            dtype: f32t
        }]
    );
}

/// One residence per operand per level: a second statement is a contradiction, not an update.
#[test]
#[should_panic(expected = "more than one stage")]
fn over_double_statement_at_one_level_panics() {
    let mut ops = matmul_operands();
    let _ = Tiling::over(&mut ops, &[(M, 64), (N, 64), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, Cut::sequential(16))
                .axis(N, Cut::sequential(16))
                .axis(K, Cut::sequential(16));
            o.a.stage(Residence::Smem);
            o.a.stage(Residence::Register);
        })
        .build();
}

/// The stages is the format statement a launch stamps: the residence column feeds the
/// [`TileSpec`](cubek_tile::TileSpec), and the finest register stage is what the operand is at
/// the instruction; one stating none is a memory window, read where it lies by whichever
/// instruction consumes it.
#[test]
fn over_records_where_each_operand_lives_and_the_space_names_the_instruction() {
    let mut ops = matmul_operands();
    let space = Tiling::over(&mut ops, &[(M, 64), (N, 64), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, Cut::sequential(16))
                .axis(N, Cut::sequential(16))
                .axis(K, Cut::sequential(16));
            o.a.stage(Residence::Smem);
        })
        .instruction(Instruction::Cmma, |l, o| {
            l.axis(M, Cut::sequential(8))
                .axis(N, Cut::sequential(8))
                .axis(K, Cut::sequential(4));
            o.a.stage(Residence::Register);
        })
        .build();

    // An operand says where it lives, and nothing about what consumes it there.
    assert_eq!(ops.a.residences(), &[Residence::Smem, Residence::Register]);
    assert!(Instruction::stages_to_registers(&ops.a.residences()));
    assert!(!Instruction::stages_to_registers(&ops.b.residences()));

    // Which instruction those registers are for is said once, by the space.
    assert_eq!(space.instruction(), Some(Instruction::Cmma));
}

// ---- A level that cuts nothing --------------------------------------------

/// A level whose edges are the extents handed to it has one instance on every axis, so it
/// partitions nothing. The build drops it, and the plan compares equal to the plan without it,
/// which is what stops the two from compiling as two kernels.
#[test]
fn a_level_that_cuts_nothing_is_dropped() {
    let plain = Tiling::new()
        .extents(&[(M, 64), (N, 64)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(16)).axis(N, Cut::sequential(32))
        })
        .build();
    let space = Tiling::new()
        .extents(&[(M, 64), (N, 64)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(16)).axis(N, Cut::sequential(32))
        })
        // The same edges again: nothing left to cut.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(16)).axis(N, Cut::sequential(32))
        })
        .build();

    assert_eq!(space, plain);
    assert_eq!(space.partitioner().depth(), 1);
}

/// A one-instance level whose pipeline is deeper than its parent's is a real stage: it buffers
/// two regions where its parent buffers one, so it is not a no-op and stays.
#[test]
fn a_level_that_cuts_nothing_but_buffers_deeper_stays() {
    let space = Tiling::new()
        .extents(&[(M, 64), (N, 64)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(16)).axis(N, Cut::sequential(32))
        })
        .level(WalkOrder::RowMajor, Buffering::DOUBLE, |l| {
            l.axis(M, Cut::sequential(16)).axis(N, Cut::sequential(32))
        })
        .build();

    assert_eq!(space.partitioner().depth(), 2);
    match space.divide().partitioner() {
        Partitioner::Level(level) => assert_eq!(level.buffering(), Buffering::DOUBLE),
        Partitioner::Final => panic!("the deeper-buffered level was dropped"),
    }
}

/// The only level of a space stays even when its cuts take the whole extent: a space with a
/// level is a tile walked in cells, and a space with none is the cell. A one-region walk is the
/// degenerate case of the first, not the second: the shape a plan takes when the knob it splits
/// on (attention's split count, a plane grid of one) lands on 1.
#[test]
fn the_only_level_stays_even_when_it_cuts_nothing() {
    let space = Tiling::new()
        .extents(&[(M, 64), (N, 64)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
            l.axis(M, Cut::sequential(64)).axis(N, Cut::sequential(64))
        })
        .build();

    assert_eq!(space.partitioner().depth(), 1);
    assert!(space.divide().is_final());
}

/// An operand moving at a one-instance level is real staging and real traffic, so the level
/// stays, and with it the residence column entry that names the move.
#[test]
fn a_level_that_cuts_nothing_but_moves_an_operand_stays() {
    let mut ops = matmul_operands();
    let space = Tiling::over(&mut ops, &[(M, 64), (N, 64), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::sequential(16))
                .axis(N, Cut::sequential(16))
                .axis(K, Cut::sequential(16));
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, Cut::sequential(16))
                .axis(N, Cut::sequential(16))
                .axis(K, Cut::sequential(16));
            o.a.stage(Residence::Smem);
        })
        .build();

    assert_eq!(space.partitioner().depth(), 2);
    assert_eq!(ops.a.residences(), &[Residence::InPlace, Residence::Smem]);
}

/// The instruction level carries the instruction, which is its whole purpose; it stays even
/// when its cuts take nothing off its parent. Every other operand's column shortens with the
/// levels that go, so it keeps one entry per level of the space that was built.
#[test]
fn the_instruction_level_stays_and_the_columns_follow_the_levels() {
    let mut ops = matmul_operands();
    let space = Tiling::over(&mut ops, &[(M, 64), (N, 64), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, o| {
            l.axis(M, Cut::sequential(16))
                .axis(N, Cut::sequential(16))
                .axis(K, Cut::sequential(16));
            o.a.stage(Residence::Smem);
        })
        // Dropped: same edges, nothing stated.
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::sequential(16))
                .axis(N, Cut::sequential(16))
                .axis(K, Cut::sequential(16));
        })
        .instruction(Instruction::Cmma, |l, _| {
            l.axis(M, Cut::sequential(16))
                .axis(N, Cut::sequential(16))
                .axis(K, Cut::sequential(16));
        })
        .build();

    assert_eq!(space.partitioner().depth(), 2);
    assert_eq!(space.instruction(), Some(Instruction::Cmma));
    assert_eq!(ops.a.residences(), &[Residence::Smem, Residence::InPlace]);
}

/// The build seals its operands: a stage stated afterwards would describe a level that does
/// not exist, silently misaligning every stage under it.
#[test]
#[should_panic(expected = "after the space was built")]
fn over_staging_after_build_panics() {
    let mut ops = matmul_operands();
    let _ = Tiling::over(&mut ops, &[(M, 64), (N, 64), (K, 16)])
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.axis(M, Cut::sequential(16))
                .axis(N, Cut::sequential(16))
                .axis(K, Cut::sequential(16));
        })
        .build();
    ops.a.stage(Residence::Smem);
}
