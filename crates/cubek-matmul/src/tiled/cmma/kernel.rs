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
    Axis, Buffering, CubeAxis, DeliveryFamily, Monoid, PlanePartition, Ring, Semiring, Space,
    StageStorage, TileArg, Tiling, Walk, WalkOrder, cubes, pipelined, planes,
};

use crate::tiled::{K, M, N, cmma::base::CmmaBlueprint};

/// The routine's four-level space in kernel form, its top extents resolved from the tensors:
/// the cube grid walking `K` in stages, one partition per plane, the instruction's `K` steps
/// through the partition, and the fragment grid each step contracts. `batch` lists the
/// surviving (extent > 1) output batch axes, one per cube on `Z`.
pub fn cmma_space(bp: &CmmaBlueprint, batch: &[Axis]) -> Space {
    let (i, c) = (bp.instruction, bp.partition);
    let (stage_m, stage_n) = bp.stage();
    let stage_k = bp.stage_k;

    // One tile of every batch axis, which is what each level states of them.
    let batch_tiles: Vec<_> = batch.iter().map(|&a| (a, 1)).collect();
    let axes: Vec<_> = batch.iter().copied().chain([M, N, K]).collect();

    Tiling::axes(&mut (), &axes)
        .level(
            WalkOrder::RowMajor,
            Buffering::new(bp.buffering),
            |l, _| {
                l.distribute(cubes(CubeAxis::Z), &batch_tiles)
                    .distribute(cubes(CubeAxis::X), &[(M, stage_m)])
                    .distribute(cubes(CubeAxis::Y), &[(N, stage_n)])
                    .walk(&[(K, stage_k)]);
            },
        )
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.distribute(planes(), &[(M, c.m * i.m)])
                .distribute(planes(), &[(N, c.n * i.n)])
                .walk(&batch_tiles)
                .walk(&[(K, stage_k)]);
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.walk(&batch_tiles)
                .walk(&[(M, c.m * i.m), (N, c.n * i.n), (K, i.k)]);
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.walk(&batch_tiles).walk(&[(M, i.m), (N, i.n), (K, i.k)]);
        })
        .build()
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
    let a = D::tile::<EL, VA>(a, comptime!(space.clone()));
    let b = D::tile::<ER, VB>(b, comptime!(space.clone()));
    let mut c = c.tile(space);

    // The accumulator spans the whole K walk: opened here, drained after it.
    let mut acc = c.cmma_accumulator::<EA, EL>(&a, Monoid::Sum);
    acc.zero();

    // This cube's box, one stage of K per region, both inputs staged for it.
    let stages = Walk::over(c.op_space(&a, &b));
    let mut ring = Ring::smem(&stages, &a, &b, StageStorage::Tiled, depth);
    pipelined(stages, &mut ring, |slot, stage| {
        let acc_stage = acc.at(stage);
        slot.consume(|a_s, b_s| {
            // This plane's box of the stage.
            for plane in Walk::over(acc_stage.op_space(a_s, b_s)) {
                let acc_plane = acc_stage.at(&plane);
                let a_p = a_s.at(&plane);
                let b_p = b_s.at(&plane);
                // The instruction's K steps through the box, the operands loaded into
                // fragments per step.
                for step in Walk::over(acc_plane.op_space(&a_p, &b_p)).unrolled() {
                    let acc_step = acc_plane.at(&step);
                    let a_f = PlanePartition::<EL>::cmma_fragments(&a_p.at(&step), &acc_step);
                    let b_f = PlanePartition::<ER>::cmma_fragments(&b_p.at(&step), &acc_step);
                    // Every fragment of the partition, contracted through the instruction.
                    for cell in Walk::over(acc_step.op_space(&a_f, &b_f)).unrolled() {
                        let mut acc_cell = acc_step.at(&cell);
                        acc_cell.mma(&a_f.at(&cell), &b_f.at(&cell), Semiring::SUM_PROD);
                    }
                }
            }
        });
    });
    acc.drain_cast_into(&mut c);
}
