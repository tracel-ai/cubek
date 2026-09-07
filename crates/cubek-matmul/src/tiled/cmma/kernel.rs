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
    Axis, Cut, DeliveryFamily, Fragments, Level, Monoid, PlanePartition, Ring, Semiring, Space,
    StageStorage, TileArg, pipelined,
};

use crate::tiled::{K, M, N, cmma::base::CmmaBlueprint};

/// The routine's five levels, each a method on the blueprint, outermost first: the cube grid,
/// the stages of `K` a cube walks, one partition per plane, the instruction's `K` steps through
/// the partition, and the fragment grid each step contracts. The kernel's loops state them one
/// by one; the launch lists them for the operand gates and checks them against the grid.
pub fn cmma_levels(bp: &CmmaBlueprint, batch: &[Axis]) -> Vec<Level> {
    vec![
        bp.cubes(batch),
        bp.k_stages(),
        bp.planes(),
        bp.k_steps(),
        bp.fragments(),
    ]
}

impl CmmaBlueprint {
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

    /// The cube grid: a box of the output per cube, one of every batch axis.
    pub fn cubes(&self, batch: &[Axis]) -> Level {
        let (stage_m, stage_n) = self.stage();
        Level::cubes(&[(M, stage_m), (N, stage_n)]).batches(batch)
    }

    /// The cube's box walked through `K` one stage at a time.
    pub fn k_stages(&self) -> Level {
        Level::walk(&[(K, self.stage_k)])
    }

    /// The stage split across the blueprint's planes, one partition each.
    pub fn planes(&self) -> Level {
        let (c, i, p) = (self.partition, self.instruction, self.planes);
        Level::planes(&[
            Cut::new(M, c.m * i.m).across(p.m),
            Cut::new(N, c.n * i.n).across(p.n),
        ])
    }

    /// The partition stepped through the stage's `K` in the instruction's depth.
    pub fn k_steps(&self) -> Level {
        Level::walk(&[(K, self.instruction.k)])
    }

    /// The partition's grid of fragments, one instruction each.
    pub fn fragments(&self) -> Level {
        let i = self.instruction;
        Level::walk(&[(M, i.m), (N, i.n)])
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
        let stages = cube.walk(comptime!(bp.k_stages()));
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
                for plane in stage.planes(comptime!(bp.planes())) {
                    // The plane's window of each stage, taken once: the slot's origin is a
                    // runtime value, so a window per step would pay its load and add per step.
                    let acc_plane = acc_stage.at(&plane);
                    let a_p = a_s.at(&plane);
                    let b_p = b_s.at(&plane);
                    // The operands are loaded into fragments one K step at a time.
                    for step in plane.walk(comptime!(bp.k_steps())).unrolled() {
                        let acc_step = acc_plane.at(&step);
                        let a_f = PlanePartition::<EL>::cmma_fragments(&a_p.at(&step), &acc_step);
                        let b_f = PlanePartition::<ER>::cmma_fragments(&b_p.at(&step), &acc_step);
                        for cell in step.walk(comptime!(bp.fragments())).unrolled() {
                            let mut acc_cell = acc_step.at(&cell);
                            acc_cell.mma(&a_f.at(&cell), &b_f.at(&cell), Semiring::SUM_PROD);
                        }
                    }
                }
            });
        });
        // Each fragment to its window of the output, cast down to its type.
        for plane in cube.planes(comptime!(bp.planes())) {
            for cell in plane.walk(comptime!(bp.fragments())).unrolled() {
                let mut c_cell = c.at(&cell);
                c_cell.copy_cast_from(&acc.at(&cell));
            }
        }
    }
}
