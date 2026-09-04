//! The CpuGemm kernel: the space it runs over and the walk written out, level by level.

use cubecl::prelude::*;
use cubek_tile::{
    Axis, Buffering, CubeAxis, Monoid, RegisterBlock, Semiring, Space, TileArg, Tiling, Walk,
    WalkOrder, cubes, planes,
};

use crate::tiled::{K, M, N, cpu_gemm::base::CpuGemmBlueprint};

/// The register block the software instruction runs under on a CPU backend: a wide scalar
/// register budget to unroll against and the dual-path edge specialization, with no lanes to
/// fan out over. Stated here because the kernel is what runs it.
pub const REGISTER_BLOCK: RegisterBlock = RegisterBlock::new(256).split_edge();

/// The routine's two-level space in kernel form: the cube grid (a serial loop on CPU) walking
/// `K` whole, then the plane split (the parallel worker threads) stepping `K` in the leaf's
/// depth. `batch` lists the surviving (extent > 1) output batch axes, one per cube on `Z`. `k`
/// is stated because the cube walks it whole in one region, which is an edge and so comptime:
/// the kernel is compiled per contraction depth, as it always was.
pub fn cpu_gemm_space(bp: &CpuGemmBlueprint, batch: &[Axis], k: usize) -> Space {
    let leaf = bp.instruction;
    let cube_m = bp.planes.m * leaf.m;
    let cube_n = bp.planes.n * leaf.n;
    let batch_tiles: Vec<_> = batch.iter().map(|&a| (a, 1)).collect();
    let axes: Vec<_> = batch.iter().copied().chain([M, N, K]).collect();

    Tiling::axes(&mut (), &axes)
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.distribute(cubes(CubeAxis::Z), &batch_tiles)
                .distribute(cubes(CubeAxis::X), &[(M, cube_m)])
                .distribute(cubes(CubeAxis::Y), &[(N, cube_n)])
                .walk(&[(K, k)]);
        })
        .level(WalkOrder::RowMajor, Buffering::SINGLE, |l, _| {
            l.distribute(planes(), &[(M, leaf.m)])
                .distribute(planes(), &[(N, leaf.n)])
                .walk(&batch_tiles)
                .walk(&[(K, leaf.k)]);
        })
        .build()
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
    #[comptime] bp: CpuGemmBlueprint,
    #[comptime] batch: Vec<Axis>,
    #[comptime] k: usize,
    #[define(EL)] _lhs_dtype: ElemType,
    #[define(ER)] _rhs_dtype: ElemType,
    #[define(E)] _acc_dtype: ElemType,
    #[define(EA)] _acc_register_dtype: ElemType,
) {
    let space = comptime!(cpu_gemm_space(&bp, &batch, k));
    let a = a.tile(comptime!(space.clone()));
    let b = b.tile(comptime!(space.clone()));
    let mut c = c.tile(space);

    // The accumulator spans the cube's whole contraction: opened here, drained after it.
    let mut acc = c.block_accumulator::<EA, EL>(&a, REGISTER_BLOCK, Monoid::Sum);
    acc.zero();

    // This cube's box, K whole: one region.
    for cube in Walk::over(c.op_space(&a, &b)) {
        let acc_cube = acc.at(&cube);
        let a_cube = a.at(&cube);
        let b_cube = b.at(&cube);
        // This plane's block, stepped through K in the instruction's depth.
        for step in Walk::over(acc_cube.op_space(&a_cube, &b_cube)) {
            let mut acc_step = acc_cube.at(&step);
            acc_step.mma(&a_cube.at(&step), &b_cube.at(&step), Semiring::SUM_PROD);
        }
    }
    acc.drain_cast_into(&mut c);
}
