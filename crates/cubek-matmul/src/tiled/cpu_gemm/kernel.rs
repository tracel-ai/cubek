//! The CpuGemm kernel: the space it runs over and the walk written out, level by level.

use cubecl::prelude::*;
use cubek_tile::{
    Axis, CubeAxis, Fragments, Level, Monoid, RegisterBlock, Semiring, Space, TileArg, cubes,
    planes,
};

use crate::tiled::{K, M, N, cpu_gemm::base::CpuGemmBlueprint};

/// The register block the software instruction runs under on a CPU backend: a wide scalar
/// register budget to unroll against and the dual-path edge specialization, with no lanes to
/// fan out over. Stated here because the kernel is what runs it.
pub const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(256).split_edge();

/// The axes of the routine's space: `batch` (the surviving, extent > 1, output batch axes) then
/// `M`, `N`, `K`.
pub fn cpu_gemm_axes(batch: &[Axis]) -> Vec<Axis> {
    batch.iter().copied().chain([M, N, K]).collect()
}

/// The routine's two levels, outermost first: the cube grid (a serial loop on CPU) walking `K`
/// whole, then the plane split (the parallel worker threads) stepping `K` in the leaf's depth.
/// `k` is stated because the cube walks it whole in one region, which is an edge and so
/// comptime: the kernel is compiled per contraction depth, as it always was.
pub fn cpu_gemm_levels(bp: &CpuGemmBlueprint, batch: &[Axis]) -> Vec<Level> {
    vec![bp.cubes(batch), bp.planes(batch), bp.steps(batch)]
}

impl CpuGemmBlueprint {
    fn batch_tiles(batch: &[Axis]) -> Vec<(Axis, usize)> {
        batch.iter().map(|&a| (a, 1)).collect()
    }

    /// The cube grid, `K` whole.
    pub fn cubes(&self, batch: &[Axis]) -> Level {
        let leaf = self.instruction;
        let cube_m = self.planes.m * leaf.m;
        let cube_n = self.planes.n * leaf.n;
        Level::new(&cpu_gemm_axes(batch), |l| {
            l.distribute(cubes(CubeAxis::Z), &Self::batch_tiles(batch))
                .distribute(cubes(CubeAxis::X), &[(M, cube_m)])
                .distribute(cubes(CubeAxis::Y), &[(N, cube_n)])
                .whole(&[K]);
        })
    }

    /// One register block per plane, `K` whole.
    pub fn planes(&self, batch: &[Axis]) -> Level {
        let leaf = self.instruction;
        Level::new(&cpu_gemm_axes(batch), |l| {
            l.distribute(planes(), &[(M, leaf.m)])
                .distribute(planes(), &[(N, leaf.n)])
                .whole(batch)
                .whole(&[K]);
        })
    }

    /// The plane's block stepped through `K` in the leaf's depth.
    pub fn steps(&self, batch: &[Axis]) -> Level {
        let leaf = self.instruction;
        Level::new(&cpu_gemm_axes(batch), |l| {
            l.whole(batch).whole(&[M, N]).walk(&[(K, leaf.k)]);
        })
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
/// The cube's box of the output is one region, its whole `K` walked at the level below by the
/// planes: each plane owns one register block and steps it through `K` in the instruction's
/// depth.
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

    // The accumulator spans the cube's whole contraction: opened here, drained after it. One
    // block per plane, the instruction's shape.
    let leaf = comptime!(bp.instruction);
    let fragments = comptime!(Fragments {
        m_tiles: 1,
        n_tiles: 1,
        m: leaf.m,
        n: leaf.n,
        k: leaf.k,
    });
    let mut acc = c.block_accumulator::<EA, EL>(&a, fragments, REGISTER_BLOCK, Monoid::Sum);
    acc.zero();

    for cube in space.cubes(comptime!(bp.cubes(&batch))) {
        let acc_cube = acc.at(&cube);
        let a_cube = a.at(&cube);
        let b_cube = b.at(&cube);
        for plane in cube.planes(comptime!(bp.planes(&batch))) {
            let acc_plane = acc_cube.at(&plane);
            let a_plane = a_cube.at(&plane);
            let b_plane = b_cube.at(&plane);
            for step in plane.walk(comptime!(bp.steps(&batch))) {
                let mut acc_step = acc_plane.at(&step);
                acc_step.mma(&a_plane.at(&step), &b_plane.at(&step), Semiring::SUM_PROD);
            }
        }
    }
    // Each plane's block to its window of the output, cast down to its type.
    for cube in space.cubes(comptime!(bp.cubes(&batch))) {
        for plane in cube.planes(comptime!(bp.planes(&batch))) {
            let mut c_plane = c.at(&plane);
            c_plane.copy_cast_from(&acc.at(&plane));
        }
    }
}
