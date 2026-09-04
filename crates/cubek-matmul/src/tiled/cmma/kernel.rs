//! The Cmma kernel: the space it runs over and the walk written out, level by level.
//!
//! [`cmma_space`] is the one statement of the space, called by the launch for the grid and by
//! the kernel for its walk, so the two cannot drift. Each level of it is a loop here, each
//! stage a ring the kernel allocates, and the accumulator a bracket the kernel opens before the
//! `K` walk and drains after it. One body serves both delivery families (strided cooperative
//! copy or TMA bulk copy; the output is always strided): the ring's pipeline is deduced from
//! what the operands are.

use cubecl::prelude::*;
use cubek_tile::{
    Axis, CubeAxis, DeliveryFamily, Fragments, Level, Monoid, PlanePartition, Ring, Semiring,
    Space, StageStorage, TileArg, cubes, pipelined, planes,
};

use crate::tiled::{K, M, N, cmma::base::CmmaBlueprint};

/// The axes of the routine's space: `batch` (the surviving, extent > 1, output batch axes) then
/// `M`, `N`, `K`, the canonical order every level is stated in.
pub fn cmma_axes(batch: &[Axis]) -> Vec<Axis> {
    batch.iter().copied().chain([M, N, K]).collect()
}

/// The routine's four levels, each a method on the blueprint, outermost first: the cube grid
/// walking `K` in stages, one partition per plane, the instruction's `K` steps through the
/// partition, and the fragment grid each step contracts. The kernel's loops state them one by
/// one and the launch lists them here, so the two cannot drift on a value.
pub fn cmma_levels(bp: &CmmaBlueprint, batch: &[Axis]) -> Vec<Level> {
    vec![
        bp.stage_level(batch),
        bp.plane_level(batch),
        bp.step_level(batch),
        bp.cell_level(batch),
    ]
}

/// The routine's space in kernel form, its top extents resolved from the tensors.
pub fn cmma_space(bp: &CmmaBlueprint, batch: &[Axis]) -> Space {
    Space::dynamic(&cmma_axes(batch)).with_levels(&cmma_levels(bp, batch))
}

impl CmmaBlueprint {
    /// One tile of every batch axis, which is what each level states of them.
    fn batch_tiles(batch: &[Axis]) -> Vec<(Axis, usize)> {
        batch.iter().map(|&a| (a, 1)).collect()
    }

    /// The cube grid: a box of the output per cube, `K` walked one stage at a time.
    pub fn stage_level(&self, batch: &[Axis]) -> Level {
        let (stage_m, stage_n) = self.stage();
        let stage_k = self.stage_k;
        Level::cuts(&cmma_axes(batch), |l| {
            l.distribute(cubes(CubeAxis::Z), &Self::batch_tiles(batch))
                .distribute(cubes(CubeAxis::X), &[(M, stage_m)])
                .distribute(cubes(CubeAxis::Y), &[(N, stage_n)])
                .walk(&[(K, stage_k)]);
        })
    }

    /// One partition per plane, the stage's `K` whole.
    pub fn plane_level(&self, batch: &[Axis]) -> Level {
        let (i, c) = (self.instruction, self.partition);
        let stage_k = self.stage_k;
        Level::cuts(&cmma_axes(batch), |l| {
            l.distribute(planes(), &[(M, c.m * i.m)])
                .distribute(planes(), &[(N, c.n * i.n)])
                .walk(&Self::batch_tiles(batch))
                .walk(&[(K, stage_k)]);
        })
    }

    /// The instruction's `K` steps through the partition.
    pub fn step_level(&self, batch: &[Axis]) -> Level {
        let (i, c) = (self.instruction, self.partition);
        Level::cuts(&cmma_axes(batch), |l| {
            l.walk(&Self::batch_tiles(batch))
                .walk(&[(M, c.m * i.m), (N, c.n * i.n), (K, i.k)]);
        })
    }

    /// The fragment grid each step contracts.
    pub fn cell_level(&self, batch: &[Axis]) -> Level {
        let i = self.instruction;
        Level::cuts(&cmma_axes(batch), |l| {
            l.walk(&Self::batch_tiles(batch))
                .walk(&[(M, i.m), (N, i.n), (K, i.k)]);
        })
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
    #[comptime] bp: CmmaBlueprint,
    #[comptime] batch: Vec<Axis>,
    #[define(EL)] _lhs_dtype: ElemType,
    #[define(ER)] _rhs_dtype: ElemType,
    #[define(E)] _acc_dtype: ElemType,
    #[define(EA)] _acc_register_dtype: ElemType,
) {
    let space = comptime!(cmma_space(&bp, &batch));
    let depth = comptime!(bp.buffering);
    let (i, c_grid) = comptime!((bp.instruction, bp.partition));
    // This plane's fragments: the partition's grid of the instruction's tile.
    let fragments = comptime!(Fragments::new(c_grid.m, c_grid.n, i.m, i.n, i.k));
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
    let mut c = c.tile(space);

    // The accumulator spans the whole K walk: opened here, drained after it.
    let mut acc = c.cmma_accumulator::<EA, EL>(&a, fragments, Monoid::Sum);
    acc.zero();

    // This cube's box, one stage of K per region, both inputs staged for it.
    let stages = c.op_space(&a, &b).level(comptime!(bp.stage_level(&batch)));
    let mut ring = Ring::smem(&stages, &a, &b, comptime!(StageStorage::tiled(&block)), depth);
    pipelined(stages, &mut ring, |slot, stage| {
        let acc_stage = acc.at(stage);
        slot.consume(|a_s, b_s| {
            // This plane's box of the stage.
            for region in acc_stage
                .op_space(a_s, b_s)
                .level(comptime!(bp.plane_level(&batch)))
            {
                let acc_plane = acc_stage.at(&region);
                let a_p = a_s.at(&region);
                let b_p = b_s.at(&region);
                // The instruction's K steps through the box, the operands loaded into
                // fragments per step.
                for step in acc_plane
                    .op_space(&a_p, &b_p)
                    .level(comptime!(bp.step_level(&batch)))
                    .unrolled()
                {
                    let acc_step = acc_plane.at(&step);
                    let a_f = PlanePartition::<EL>::cmma_fragments(&a_p.at(&step), &acc_step);
                    let b_f = PlanePartition::<ER>::cmma_fragments(&b_p.at(&step), &acc_step);
                    // Every fragment of the partition, contracted through the instruction.
                    for cell in acc_step
                        .op_space(&a_f, &b_f)
                        .level(comptime!(bp.cell_level(&batch)))
                        .unrolled()
                    {
                        let mut acc_cell = acc_step.at(&cell);
                        acc_cell.mma(&a_f.at(&cell), &b_f.at(&cell), Semiring::SUM_PROD);
                    }
                }
            }
        });
    });
    acc.drain_cast_into(&mut c);
}
