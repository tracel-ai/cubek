//! The CpuGemm kernel: the space it runs over and the walk written out, level by level.

use cubecl::prelude::*;
use cubek_tile::{
    Accumulate, AccumulateExpand, Axis, Level, Levels, Monoid, Partitioning, RegisterBlock,
    Semiring, Space, TileArg,
};

use crate::tiled::{K, M, N, cpu_gemm::base::CpuGemmBlueprint};

/// The register block the software instruction runs under on a CPU backend: a wide scalar
/// register budget to unroll against and the dual-path edge specialization, with no lanes to
/// fan out over. Stated here because the kernel is what runs it.
pub const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(256).split_edge();

/// The routine's three levels, stated **from the leaf up, in counts**: the register block,
/// every step of `K` in its depth, the planes a cube holds (the parallel worker threads), and a
/// cube per box of the output (a serial loop on CPU). The kernel's loops state them one by one
/// and read their leaf and overhangs off the same list.
pub fn cpu_gemm_levels(bp: &CpuGemmBlueprint, batch: &[Axis]) -> Vec<Level> {
    let (leaf, p) = (bp.instruction, bp.planes);
    Levels::leaf(&[(M, leaf.m), (N, leaf.n), (K, leaf.k)])
        .walk_every(&[K])
        .planes(&[(M, p.m), (N, p.n)])
        .cubes(&[M, N])
        .batches(batch)
        .build()
}

impl CpuGemmBlueprint {
    /// The space with the levels that cut it: what the leaf and the overhangs are read off.
    pub fn partitioning(&self, space: &Space, batch: &[Axis]) -> Partitioning {
        Partitioning::new(space.clone(), cpu_gemm_levels(self, batch))
    }

    /// The grid this launch runs on: a cube per box of the output and per batch, the blueprint's
    /// planes in each.
    pub fn grid(&self, space: &Space, batch: &[Axis], plane_size: u32) -> (CubeCount, CubeDim) {
        let leaf = self.instruction;
        let (cube_m, cube_n) = (self.planes.m * leaf.m, self.planes.n * leaf.n);
        let batches: usize = batch.iter().map(|&a| space.extent(a)).product();
        (
            CubeCount::Static(
                space.extent(M).div_ceil(cube_m) as u32,
                space.extent(N).div_ceil(cube_n) as u32,
                batches as u32,
            ),
            CubeDim::new_2d(plane_size, (self.planes.m * self.planes.n) as u32),
        )
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
    space: Partitioning,
    #[define(EL)] _lhs_dtype: ElemType,
    #[define(ER)] _rhs_dtype: ElemType,
    #[define(E)] _acc_dtype: ElemType,
    #[define(EA)] _acc_register_dtype: ElemType,
) {
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let c = c.tile(comptime!(space.clone()));

    for cube in space {
        for plane in cube {
            let a = a.at(&plane);
            let b = b.at(&plane);
            let mut c = c.at(&plane);
            // One block per plane, the instruction's shape, read off the levels below the plane.
            let mut acc = c.block_accumulator::<EA, EL, ER>(&a, &b, REGISTER_BLOCK, Monoid::Sum);
            acc.zero();
            for step in plane {
                let mut acc_step = acc.at(&step);
                acc_step.mma(&a.at(&step), &b.at(&step), Semiring::SUM_PROD);
            }
            c.copy_cast_from(&acc);
        }
    }
}
