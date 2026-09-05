//! The Cmma kernel: its levels, one method of the blueprint each, and the walk written out,
//! level by level.
//!
//! The launch lists the level methods for the grid and hands the kernel its space; each loop
//! here names the level it walks, so the two cannot drift. Each stage is a ring the kernel
//! allocates, and the accumulator a bracket the kernel opens before the `K` walk and drains
//! after it. One body serves both delivery families (strided cooperative
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
        bp.cubes(batch),
        bp.stages(batch),
        bp.planes(batch),
        bp.steps(batch),
        bp.cells(batch),
    ]
}

impl CmmaBlueprint {
    /// One tile of every batch axis, which is what each level states of them.
    fn batch_tiles(batch: &[Axis]) -> Vec<(Axis, usize)> {
        batch.iter().map(|&a| (a, 1)).collect()
    }

    /// The cube grid: a box of the output per cube, `K` whole.
    pub fn cubes(&self, batch: &[Axis]) -> Level {
        let (stage_m, stage_n) = self.stage();
        Level::new(&cmma_axes(batch), |l| {
            l.distribute(cubes(CubeAxis::Z), &Self::batch_tiles(batch))
                .distribute(cubes(CubeAxis::X), &[(M, stage_m)])
                .distribute(cubes(CubeAxis::Y), &[(N, stage_n)])
                .whole(&[K]);
        })
    }

    /// The cube's box walked along `K` one stage at a time.
    pub fn stages(&self, batch: &[Axis]) -> Level {
        let stage_k = self.stage_k;
        Level::new(&cmma_axes(batch), |l| {
            l.whole(batch).whole(&[M, N]).walk(&[(K, stage_k)]);
        })
    }

    /// One partition per plane, the stage's `K` whole.
    pub fn planes(&self, batch: &[Axis]) -> Level {
        let (i, c) = (self.instruction, self.partition);
        Level::new(&cmma_axes(batch), |l| {
            l.distribute(planes(), &[(M, c.m * i.m)])
                .distribute(planes(), &[(N, c.n * i.n)])
                .whole(batch)
                .whole(&[K]);
        })
    }

    /// The instruction's `K` steps through the partition.
    pub fn steps(&self, batch: &[Axis]) -> Level {
        let i = self.instruction;
        Level::new(&cmma_axes(batch), |l| {
            l.whole(batch).whole(&[M, N]).walk(&[(K, i.k)]);
        })
    }

    /// The fragment grid each step contracts.
    pub fn cells(&self, batch: &[Axis]) -> Level {
        let i = self.instruction;
        Level::new(&cmma_axes(batch), |l| {
            l.whole(batch).whole(&[K]).walk(&[(M, i.m), (N, i.n)]);
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
    space: Space,
    #[comptime] bp: CmmaBlueprint,
    #[comptime] batch: Vec<Axis>,
    #[define(EL)] _lhs_dtype: ElemType,
    #[define(ER)] _rhs_dtype: ElemType,
    #[define(E)] _acc_dtype: ElemType,
    #[define(EA)] _acc_register_dtype: ElemType,
) {
    let depth = comptime!(bp.buffering);
    let (i, c_grid) = comptime!((bp.instruction, bp.partition));
    // This plane's fragments: the partition's grid of the instruction's tile.
    let fragments = comptime!(Fragments {
        m_tiles: c_grid.m,
        n_tiles: c_grid.n,
        m: i.m,
        n: i.n,
        k: i.k,
    });
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

    for cube in space.cubes(comptime!(bp.cubes(&batch))) {
        let a = a.at(&cube);
        let b = b.at(&cube);
        let c = c.at(&cube);
        // The accumulator spans the whole K walk: opened here, drained after it.
        let mut acc = c.cmma_accumulator::<EA, EL>(&a, fragments, Monoid::Sum);
        acc.zero();

        // One stage of K per region, both inputs staged for it.
        let stages = cube.walk(comptime!(bp.stages(&batch)));
        let mut ring = Ring::smem(
            &stages,
            &a,
            &b,
            comptime!(StageStorage::Tiled {
                block: block.clone()
            }),
            depth,
        );
        pipelined(stages, &mut ring, |slot, stage| {
            let acc_stage = acc.at(stage);
            slot.consume(|a_s, b_s| {
                for plane in stage.planes(comptime!(bp.planes(&batch))) {
                    // The plane's window of each stage, taken once: the slot's origin is a
                    // runtime value, so a window per step would pay its load and add per step.
                    let acc_plane = acc_stage.at(&plane);
                    let a_p = a_s.at(&plane);
                    let b_p = b_s.at(&plane);
                    // The operands are loaded into fragments one K step at a time.
                    for step in plane.walk(comptime!(bp.steps(&batch))).unrolled() {
                        let acc_step = acc_plane.at(&step);
                        let a_f = PlanePartition::<EL>::cmma_fragments(&a_p.at(&step), &acc_step);
                        let b_f = PlanePartition::<ER>::cmma_fragments(&b_p.at(&step), &acc_step);
                        for cell in step.walk(comptime!(bp.cells(&batch))).unrolled() {
                            let mut acc_cell = acc_step.at(&cell);
                            acc_cell.mma(&a_f.at(&cell), &b_f.at(&cell), Semiring::SUM_PROD);
                        }
                    }
                }
            });
        });
        // Each fragment to its window of the output, cast down to its type.
        for plane in cube.planes(comptime!(bp.planes(&batch))) {
            for cell in plane.walk(comptime!(bp.cells(&batch))).unrolled() {
                let mut c_cell = c.at(&cell);
                c_cell.copy_cast_from(&acc.at(&cell));
            }
        }
    }
}
