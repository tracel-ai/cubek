//! The CpuGemm kernel: the space it runs over and the walk written out, level by level.

use cubecl::prelude::*;
use cubek_tile::{Axis, Cut, Fragments, Level, Monoid, RegisterBlock, Semiring, Space, TileArg};

use crate::tiled::{K, M, N, cpu_gemm::base::CpuGemmBlueprint};

/// The register block the software instruction runs under on a CPU backend: a wide scalar
/// register budget to unroll against and the dual-path edge specialization, with no lanes to
/// fan out over. Stated here because the kernel is what runs it.
pub const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(256).split_edge();

/// The routine's three levels, outermost first: the cube grid (a serial loop on CPU), the plane
/// split (the parallel worker threads), and the plane's block stepped through `K` in the
/// instruction's depth. The kernel's loops state them one by one and the launch lists them here,
/// so the two cannot drift on a value.
pub fn cpu_gemm_levels(bp: &CpuGemmBlueprint, batch: &[Axis]) -> Vec<Level> {
    vec![bp.cubes(batch), bp.planes(), bp.k_steps()]
}

impl CpuGemmBlueprint {
    /// The cube grid: a box of the output per cube, one of every batch axis.
    pub fn cubes(&self, batch: &[Axis]) -> Level {
        let leaf = self.instruction;
        let cube_m = self.planes.m * leaf.m;
        let cube_n = self.planes.n * leaf.n;
        Level::cubes(&[(M, cube_m), (N, cube_n)]).batches(batch)
    }

    /// The cube's box across the blueprint's planes, one register block each.
    pub fn planes(&self) -> Level {
        let (leaf, p) = (self.instruction, self.planes);
        Level::planes(&[
            Cut::new(M, leaf.m).across(p.m),
            Cut::new(N, leaf.n).across(p.n),
        ])
    }

    /// The plane's block stepped through `K` in the instruction's depth.
    pub fn k_steps(&self) -> Level {
        Level::walk(&[(K, self.instruction.k)])
    }
}

/// `c = a · b` in register blocks.
///
/// Each operand arrives as one argument bundling its tensor with its comptime spec; the one
/// kernel [`Space`] is built here and projected onto each in the first lines. `a` stays scalar
/// (broadcast per `K`); `b` and `c` carry the launch-chosen line size along their contiguous `N`
/// axis. Each keeps its own element type, `EL`/`ER` for the inputs, `EA` for the accumulator and
/// `E` for the stored output, and the leaf casts the inputs into `EA`, so mixed-precision GEMM
/// falls out of one kernel (same dtype is the `EL = ER = EA = E` case, where the casts fold away).
///
/// Each cube takes a box of the output; each of its planes a register block of that box, which
/// it opens, steps through `K` in the instruction's depth, and stores to its window of the
/// output, cast down to its type.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn cpu_gemm_kernel<
    E: Numeric,
    EA: Numeric,
    EL: Numeric,
    ER: Numeric,
    VA: Size,
    VB: Size,
    VC: Size,
>(
    a: &TileArg<'_, EL, VA>,
    b: &TileArg<'_, ER, VB>,
    c: &TileArg<'_, E, VC>,
    space: Space,
    #[comptime] bp: CpuGemmBlueprint,
    #[comptime] batch: Vec<Axis>,
    #[define(EL)] _lhs_dtype: ElemType,
    #[define(ER)] _rhs_dtype: ElemType,
    #[define(E)] _acc_dtype: ElemType,
    #[define(EA)] _acc_register_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));

    // One block per plane, the instruction's shape.
    let leaf = comptime!(bp.instruction);
    let fragments = comptime!(Fragments {
        m_tiles: 1,
        n_tiles: 1,
        m: leaf.m,
        n: leaf.n,
        k: leaf.k,
    });

    for cube in space.cubes(comptime!(bp.cubes(&batch))) {
        for plane in cube.planes(comptime!(bp.planes())) {
            let a = a.at(&plane);
            let b = b.at(&plane);
            let mut c = c.at(&plane);
            let mut acc = c.block_accumulator::<EA, EL>(&a, fragments, REGISTER_BLOCK, Monoid::Sum);
            acc.zero();
            for step in plane.walk(comptime!(bp.k_steps())) {
                let mut acc_step = acc.at(&step);
                acc_step.mma(&a.at(&step), &b.at(&step), Semiring::SUM_PROD);
            }
            c.copy_cast_from(&acc);
        }
    }
}
