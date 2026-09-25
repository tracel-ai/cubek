//! The Cmma kernel: its levels, one method of the blueprint each, and the walk written out,
//! level by level.
//!
//! The launch lists the level methods for the grid and hands the kernel its space; each loop
//! here names the level it walks, so the two cannot drift. Each stage is a slot the kernel
//! allocates, and the accumulator a bracket the kernel opens before the `K` walk and drains
//! after it. One body serves both delivery families (strided cooperative
//! copy or TMA bulk copy; the output is always strided): the stages' pipeline is deduced from
//! what the operands are.

use cubecl::prelude::*;
use cubek_tile::{
    Accumulate, AccumulateExpand, Axis, DeliveryFamily, Level, Levels, Monoid, Partitioning,
    PlanePartition, RowChunks, Semiring, Space, StageStorage, Stages, TileArg,
};

use crate::tiled::{K, M, N, cmma::base::CmmaBlueprint};

/// The routine's five levels, stated **from the leaf up, in counts**, outermost first once
/// built: the instruction's shape, the fragments a partition holds, the instruction steps in one
/// stage, the planes a cube holds, every stage `K` holds, and a cube per box of the output. The
/// kernel's loops state them one by one, and the two level methods it names beside its loops
/// ([`planes`](CmmaBlueprint::planes), [`fragments`](CmmaBlueprint::fragments)) read this same
/// list, so the two cannot drift.
pub fn cmma_levels(bp: &CmmaBlueprint, batch: &[Axis]) -> Vec<Level> {
    let (c, i, p) = (bp.partition, bp.instruction, bp.planes);
    Levels::leaf(&[(M, i.m), (N, i.n), (K, i.k)])
        // The partition's grid of fragments, one instruction each.
        .walk(&[(M, c.m), (N, c.n)])
        // The instruction's `K` steps through the stage.
        .walk(&[(K, bp.stage_k / i.k)])
        // The stage split across the planes, one partition each.
        .planes(&[(M, p.m), (N, p.n)])
        // The cube's box walked through `K` one stage at a time.
        .walk_every(&[K])
        // A box of the output per cube, one of every batch axis, taken in the order the
        // blueprint states.
        .cubes(&[M, N])
        .ordered(bp.order)
        .batches(batch)
        .build()
}

impl CmmaBlueprint {
    /// The space with the levels that cut it: what the leaf and the overhangs are read off.
    pub fn partitioning(&self, space: &Space, batch: &[Axis]) -> Partitioning {
        Partitioning::new(space.clone(), cmma_levels(self, batch))
    }

    /// The grid this launch runs on: a cube per stage of the output and per batch, the
    /// blueprint's planes in each.
    pub fn grid(&self, space: &Space, batch: &[Axis], plane_size: u32) -> (CubeCount, CubeDim) {
        let (stage_m, stage_n) = self.stage();
        let batches: usize = batch.iter().map(|&a| space.extent(a)).product();
        (
            CubeCount::Static(
                space.extent(M).div_ceil(stage_m) as u32,
                space.extent(N).div_ceil(stage_n) as u32,
                batches as u32,
            ),
            CubeDim::new_2d(plane_size, (self.planes.m * self.planes.n) as u32),
        )
    }

    /// The stage split across the planes, one partition each: the level the drain names
    /// beside its loop, read off the list rather than stated twice.
    pub fn planes(&self) -> Level {
        cmma_levels(self, &[])[2].clone()
    }

    /// The partition's grid of fragments, one instruction each: likewise.
    pub fn fragments(&self) -> Level {
        cmma_levels(self, &[])[4].clone()
    }
}

/// `c = a · b` on tensor cores.
///
/// The cube's box of the output is walked along `K` one stage at a time, both inputs staged into
/// shared memory for each and buffered `bp.buffering` deep so a stage's fill overlaps the previous
/// one's contraction. Inside a stage each plane takes its box, loads the instruction's operands
/// into fragments one `K` step at a time, and contracts every fragment of its partition. The
/// accumulator is resident in `EA` (typically `f32`) for the whole walk and cast down to the
/// output `E` on drain.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cmma_kernel<
    E: Numeric,
    EA: Numeric,
    EL: Numeric,
    ER: Numeric,
    VA: Size,
    VB: Size,
    VC: Size,
    D: DeliveryFamily,
>(
    a: &D::Arg<EL, VA>,
    b: &D::Arg<ER, VB>,
    c: &TileArg<'_, E, VC>,
    space: Partitioning,
    #[comptime] bp: CmmaBlueprint,
    #[comptime] batch: Vec<Axis>,
    #[define(EL)] _lhs_dtype: ElemType,
    #[define(ER)] _rhs_dtype: ElemType,
    #[define(E)] _acc_dtype: ElemType,
    #[define(EA)] _acc_register_dtype: ElemType,
) {
    let depth = comptime!(bp.buffering);
    let i = comptime!(bp.instruction);
    // The block a tiled stage groups: the instruction's tile, one of every batch axis.
    let block = comptime!(
        batch
            .iter()
            .map(|&a| (a, 1))
            .chain([(M, i.m), (N, i.n), (K, i.k)])
            .collect::<Vec<_>>()
    );
    let a = D::tile::<EL, VA>(a, comptime!(space.clone()));
    let b = D::tile::<ER, VB>(b, comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));

    for cube in space {
        let a = a.at(&cube);
        let b = b.at(&cube);
        let c = c.at(&cube);
        // The accumulator spans the whole K walk: opened here, drained after it.
        // The accumulator's grid is the partition's, read off the levels below the cube.
        let mut acc = c.cmma_accumulator::<EA, EL>(&a, Monoid::Sum);
        acc.zero();

        // One stage of K per region, both inputs staged for it.
        let steps = cube.walk();
        let mut stages = Stages::smem(
            &steps,
            &a,
            &b,
            comptime!(StageStorage::Tiled {
                block: block.clone(),
                chunks: RowChunks::InOrder,
            }),
            depth,
        );
        stages.pipelined(steps, |slot, stage| {
            let acc_stage = acc.at(stage);
            slot.consume(|a_s, b_s| {
                for plane in stage {
                    // The plane's window of each stage, taken once: the slot's origin is a
                    // runtime value, so a window per step would pay its load and add per step.
                    let acc_plane = acc_stage.at(&plane);
                    let a_p = a_s.at(&plane);
                    let b_p = b_s.at(&plane);
                    // The operands are loaded into fragments one K step at a time.
                    for step in plane.walk().unrolled() {
                        let acc_step = acc_plane.at(&step);
                        let a_f = PlanePartition::<EL>::cmma_fragments(&a_p.at(&step), &acc_step);
                        let b_f = PlanePartition::<ER>::cmma_fragments(&b_p.at(&step), &acc_step);
                        for cell in step.walk().unrolled() {
                            let mut acc_cell = acc_step.at(&cell);
                            acc_cell.mma(&a_f.at(&cell), &b_f.at(&cell), Semiring::SUM_PROD);
                        }
                    }
                }
            });
        });
        // Each fragment to its window of the output, cast down to its type: the planes and
        // their fragments, skipping the `K` levels between them, which the output does not span.
        for plane in cube.over(&bp.planes()) {
            for cell in plane.over(&bp.fragments()).unrolled() {
                let mut c_cell = c.at(&cell);
                c_cell.copy_cast_from(&acc.at(&cell));
            }
        }
    }
}

/// COUNTS_PLAN.md phase 2's gate for this kernel: the grid the leaf-up levels state is the grid
/// the blueprint counts on its own. The two were written independently, which is what makes
/// the agreement worth pinning — the old level methods stated sizes, `grid` still divides the
/// space by them, and the leaf-up levels have to land on the same cubes and planes.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tiled::cmma::{CmmaDelivery, Partition};
    use crate::tiled::cpu_gemm::{InstructionShape, PlaneGrid};
    use crate::tiled::{batch_axis, form_space, labels};

    /// A `16x16x16` instruction, `2x2` fragments a plane, `2x2` planes a cube, stages `32` deep.
    fn blueprint() -> CmmaBlueprint {
        CmmaBlueprint {
            instruction: InstructionShape {
                m: 16,
                n: 16,
                k: 16,
            },
            partition: Partition { m: 2, n: 2 },
            planes: PlaneGrid { m: 2, n: 2 },
            stage_k: 32,
            buffering: 2,
            delivery: CmmaDelivery::Copy,
            order: cubek_tile::CubeOrder::RowMajor,
        }
    }

    #[test]
    fn the_leaf_up_levels_state_the_grid_the_blueprint_counts() {
        let space = Space::new(&[(M, 256), (N, 192), (K, 480)]);
        for (partition, planes, stage_k) in [
            (Partition { m: 1, n: 1 }, PlaneGrid { m: 2, n: 1 }, 48),
            (Partition { m: 2, n: 4 }, PlaneGrid { m: 2, n: 2 }, 32),
            (Partition { m: 4, n: 2 }, PlaneGrid { m: 1, n: 4 }, 16),
        ] {
            let bp = CmmaBlueprint {
                instruction: InstructionShape { m: 8, n: 8, k: 8 },
                partition,
                planes,
                stage_k,
                buffering: 2,
                delivery: CmmaDelivery::Copy,
                order: cubek_tile::CubeOrder::RowMajor,
            };
            let partitioning = bp.partitioning(&space, &[]);
            let (count, dim) = bp.grid(&space, &[], 32);
            assert_eq!(
                format!("{:?}", partitioning.cube_count()),
                format!("{count:?}"),
                "{bp:?}"
            );
            assert_eq!(partitioning.planes_per_cube(), dim.y, "{bp:?}");
        }
    }

    /// The five levels as a table, leaf up: each row's tile is the row below it times the count
    /// beside it, so a tiling that stops dividing an axis where it meant to keep going shows up
    /// as a row that no longer multiplies out.
    #[test]
    fn the_cmma_routine_states_five_levels() {
        let batch = [batch_axis(0)];
        let space = Space::new(&[(batch[0], 4), (M, 512), (N, 1024), (K, 4096)]);
        let partitioning = blueprint().partitioning(&space, &batch);

        assert_eq!(
            partitioning.table(&labels(&space)).to_string(),
            [
                "                        b0 × m ×  n ×   k    b0 ×   m ×    n ×    k",
                "",
                "  ◦                      · × · ×  · ×   ·     1 ×  16 ×   16 ×   16",
                "  ↻  4 steps             · × 2 ×  2 ×   ·     1 ×  32 ×   32 ×   16",
                "  ↻  2 steps             · × · ×  · ×   2     1 ×  32 ×   32 ×   32",
                "  ▤  4 planes a cube     · × 2 ×  2 ×   ·     1 ×  64 ×   64 ×   32",
                "  ↻  128 steps           · × · ×  · × 128     1 ×  64 ×   64 × 4096",
                "  ▣  512 cubes           4 × 8 × 16 ×   ·     4 × 512 × 1024 × 4096",
                "",
                "                        └─ count ───────┘    └─ tile ─────────────┘",
            ]
            .join("\n")
        );
    }

    /// The kernel form: the same blueprint over a space whose extents the launch stamps. The
    /// leaf is still a number, because every level states its own tile, and only the counts an
    /// extent decides wait for a shape.
    #[test]
    fn a_dynamic_space_prints_the_form_without_the_shape() {
        let batch = [batch_axis(0)];
        let space = form_space(batch.len());
        let partitioning = blueprint().partitioning(&space, &batch);

        assert_eq!(partitioning.leaf().extent(M), 16);
        assert!(
            partitioning
                .table(&labels(&space))
                .to_string()
                .contains('?')
        );
    }
}
