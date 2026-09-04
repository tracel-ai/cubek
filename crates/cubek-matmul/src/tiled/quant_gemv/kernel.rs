//! The quantized decode gemv kernel: the space it runs over and the walk written out.

use cubecl::prelude::*;
use cubek_tile::{
    Axis, CubeAxis, Level, Region, RegisterBlock, Semiring, Space, Tile, TileArg, Walk, cubes,
    lanes, planes,
};

use crate::tiled::{
    M, N,
    quant_gemv::base::{QuantGemvBlueprint, QuantGemvProblem},
};

/// Which block of the contraction, and where inside it. Together they are `K`.
///
/// Numbered past [`batch_axis`](crate::tiled::batch_axis)'s first slots because a gemv states
/// no batch axes: the labels only have to be distinct within the space that uses them.
pub(super) const KB: cubek_tile::Axis = cubek_tile::Axis(16);
pub(super) const KI: cubek_tile::Axis = cubek_tile::Axis(17);

/// The routine's three-level space, every axis static: the extents fold the address arithmetic
/// to constants, which is what a plan built out of the shape is for. A strip of output rows per
/// cube walking all of `K`, one plane per group of rows, then the fold: `rows_per_lane` rows per
/// aligned lane group, the group's lanes interleaving the contraction between them. Each takes
/// one stored word of `KI`, and where a group reaches past one block it takes whole blocks of
/// `KB` (a distribution deals one axis or the other and cannot straddle two). The partials the
/// lanes hold drain inside the plane.
pub fn quant_gemv_space(bp: &QuantGemvBlueprint, problem: &QuantGemvProblem) -> Space {
    Space::new(&quant_gemv_extents(problem)).with_levels(&quant_gemv_levels(bp, problem))
}

/// The routine's axes and their extents, every one static.
pub fn quant_gemv_extents(problem: &QuantGemvProblem) -> Vec<(Axis, usize)> {
    vec![
        (M, problem.d_out),
        (N, problem.rows),
        (KB, problem.blocks()),
        (KI, problem.block),
    ]
}

const AXES: [Axis; 4] = [M, N, KB, KI];

/// The routine's three levels, outermost first, each a method on the blueprint.
pub fn quant_gemv_levels(bp: &QuantGemvBlueprint, problem: &QuantGemvProblem) -> Vec<Level> {
    vec![
        bp.cube_level(problem),
        bp.plane_level(problem),
        bp.lane_level(problem),
    ]
}

impl QuantGemvBlueprint {
    /// A strip of output rows per cube, all of `K` walked.
    pub fn cube_level(&self, problem: &QuantGemvProblem) -> Level {
        let (block, blocks) = (problem.block, problem.blocks());
        Level::cuts(&AXES, |l| {
            l.distribute(cubes(CubeAxis::X), &[(M, self.rows_per_cube)])
                .walk(&[(N, problem.rows), (KB, blocks), (KI, block)]);
        })
    }

    /// One plane per group of rows.
    pub fn plane_level(&self, problem: &QuantGemvProblem) -> Level {
        let (block, blocks) = (problem.block, problem.blocks());
        Level::cuts(&AXES, |l| {
            l.distribute(planes(), &[(M, self.rows_per_plane)]).walk(&[
                (N, problem.rows),
                (KB, blocks),
                (KI, block),
            ]);
        })
    }

    /// The fold: `rows_per_lane` rows per aligned lane group, the group's lanes interleaving the
    /// contraction between them. Interleaved on `(KB, KI)`, so the lanes of a group read
    /// neighbouring words. The lane counts are the blueprint's, derived on the host from the
    /// plane width: their product with the row groups is exactly it.
    pub fn lane_level(&self, problem: &QuantGemvProblem) -> Level {
        let factor = problem.factor();
        Level::cuts(&AXES, |l| {
            l.distribute(lanes(self.groups()), &[(M, self.rows_per_lane)])
                .distribute(lanes(self.block_lanes).interleaved(), &[(KB, 1)])
                .distribute(lanes(self.inside_lanes).interleaved(), &[(KI, factor)])
                .walk(&[(N, problem.rows)]);
        })
    }
}

/// The register block the leaf runs under: one scalar accumulator per row a lane owns, per
/// value of the word it takes a step.
pub fn register_block(bp: &QuantGemvBlueprint, problem: &QuantGemvProblem) -> RegisterBlock {
    RegisterBlock::new(bp.rows_per_lane * problem.factor())
}

/// `y = (W ⊗ s) · x`.
///
/// The weight arrives as `u32` words and unpacks at the read ([`TileArg::tile_packed`]); the
/// scales arrive as their own tensor at their own element type and fold in at the contraction
/// ([`Tile::mma_scaled_with`]). Nothing here mentions a quantization scheme, a block size or a
/// scale binding riding the weight: which values one scale covers is the scales operand's own
/// axes, stated in the space.
///
/// Every operand keeps its own element: `EC` is what the words decode to, `EX` what the
/// activation buffer holds, `ES` the scales', `EO` the output's, and the leaf casts each into
/// the accumulator as it always does. The activation is served in `VX`-wide lines along the
/// contraction, one stored word's worth a step, and the output scalar, because each lane holds
/// a partial of its group's cell.
///
/// Three levels, each one region per instance: the cube's strip of rows, the plane's group of
/// rows, and the lane's rows against its share of the contraction, which the leaf folds across
/// the plane's lanes as it writes.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn quant_gemv_kernel<EC: Numeric, EX: Numeric, ES: Numeric, EO: Numeric, VX: Size, VO: Size>(
    w: &TileArg<'_, u32, Const<1>>,
    x: &TileArg<'_, EX, VX>,
    scales: &Sequence<TileArg<'_, ES, Const<1>>>,
    out: &TileArg<'_, EO, VO>,
    #[comptime] bp: QuantGemvBlueprint,
    #[comptime] problem: QuantGemvProblem,
    #[define(EC)] _served_dtype: ElemType,
    #[define(EX)] _x_dtype: ElemType,
    #[define(ES)] _scale_dtype: ElemType,
    #[define(EO)] _out_dtype: ElemType,
) {
    let space = comptime!(quant_gemv_space(&bp, &problem));
    let config = comptime!(register_block(&bp, &problem));
    let w = w.tile_packed::<EC>(comptime!(space.clone()));
    let x = x.tile(comptime!(space.clone()));
    let mut scale_tiles = Sequence::new();
    #[unroll]
    for k in 0..scales.len() {
        scale_tiles.push(scales.index(k).tile(comptime!(space.clone())));
    }
    let mut out = out.tile(space);
    // The output folds every step into what it holds, so it starts from zero.
    out.zero();

    // This cube's strip of rows.
    for region in Walk::over(out.op_space(&w, &x)) {
        let out_cube = out.at(&region);
        let w_cube = w.at(&region);
        let x_cube = x.at(&region);
        let scales_cube = at_all(&scale_tiles, &region);
        // This plane's group of rows.
        for region in Walk::over(out_cube.op_space(&w_cube, &x_cube)) {
            let out_plane = out_cube.at(&region);
            let w_plane = w_cube.at(&region);
            let x_plane = x_cube.at(&region);
            let scales_plane = at_all(&scales_cube, &region);
            // This lane's rows against its share of the contraction.
            for region in Walk::over(out_plane.op_space(&w_plane, &x_plane)) {
                let mut out_lane = out_plane.at(&region);
                let scales_lane = at_all(&scales_plane, &region);
                out_lane.mma_scaled_with(
                    &w_plane.at(&region),
                    &x_plane.at(&region),
                    &scales_lane,
                    config,
                    Semiring::SUM_PROD,
                );
            }
        }
    }
}

/// Every scale level windowed to `region`.
#[cube]
fn at_all<ES: Numeric>(scales: &Sequence<Tile<ES>>, region: &Region) -> Sequence<Tile<ES>> {
    let mut at = Sequence::new();
    #[unroll]
    for k in 0..scales.len() {
        at.push(scales.index(k).at(region));
    }
    at
}
