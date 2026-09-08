//! The quantized decode gemv kernel: the space it runs over and the walk written out.

use cubecl::prelude::*;
use cubek_tile::{
    Axis, Cut, Level, Partitioning, Region, RegisterBlock, Semiring, Space, Tile, TileArg,
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
/// `KB` (a distribution cuts one axis or the other and cannot straddle two). The partials the
/// lanes hold drain inside the plane.
pub fn quant_gemv_space(problem: &QuantGemvProblem) -> Space {
    Space::new(&[
        (M, problem.d_out),
        (N, problem.rows),
        (KB, problem.blocks()),
        (KI, problem.block),
    ])
}

/// The routine's three levels, outermost first, each a method on the blueprint.
pub fn quant_gemv_levels(bp: &QuantGemvBlueprint, problem: &QuantGemvProblem) -> Vec<Level> {
    vec![bp.cubes(), bp.planes(), bp.lanes(problem)]
}

impl QuantGemvBlueprint {
    /// The space with the levels that cut it: what the leaf and the overhangs are read off.
    pub fn partitioning(&self, problem: &QuantGemvProblem) -> Partitioning {
        Partitioning::new(quant_gemv_space(problem), quant_gemv_levels(self, problem))
    }

    /// The tile every operand is cut to at the bottom: a lane's rows against one stored word.
    pub fn leaf(&self, problem: &QuantGemvProblem) -> Vec<(Axis, usize)> {
        self.partitioning(problem).leaf().extents()
    }

    /// The axes some tile reaches past the end of: none, the blueprint refuses a problem its
    /// tiles do not divide.
    pub fn overhangs(&self, problem: &QuantGemvProblem) -> Vec<Axis> {
        self.partitioning(problem).overhanging()
    }

    /// The grid this launch runs on: a cube per strip of rows, a plane per group of them, every
    /// lane of the plane.
    pub fn grid(&self, problem: &QuantGemvProblem, plane_size: u32) -> (CubeCount, CubeDim) {
        (
            CubeCount::Static(problem.d_out.div_ceil(self.rows_per_cube) as u32, 1, 1),
            CubeDim::new_2d(
                plane_size,
                (self.rows_per_cube / self.rows_per_plane) as u32,
            ),
        )
    }

    /// A strip of output rows per cube, `K` whole.
    pub fn cubes(&self) -> Level {
        Level::cubes(&[(M, self.rows_per_cube)])
    }

    /// One plane per group of rows, `K` whole.
    pub fn planes(&self) -> Level {
        Level::planes(&[(M, self.rows_per_plane)])
    }

    /// The fold: `rows_per_lane` rows per aligned lane group, the group's lanes interleaving the
    /// contraction between them. Interleaved on `(KB, KI)`, so the lanes of a group read
    /// neighbouring words. The lane counts are the blueprint's, derived on the host from the
    /// plane width: their product with the row groups is exactly it.
    pub fn lanes(&self, problem: &QuantGemvProblem) -> Level {
        Level::lanes(&[
            Cut::new(M, self.rows_per_lane).across(self.groups()),
            Cut::new(KB, 1).across(self.block_lanes).interleaved(),
            Cut::new(KI, problem.factor())
                .across(self.inside_lanes)
                .interleaved(),
        ])
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
    space: Space,
    #[comptime] bp: QuantGemvBlueprint,
    #[comptime] problem: QuantGemvProblem,
    #[define(EC)] _served_dtype: ElemType,
    #[define(EX)] _x_dtype: ElemType,
    #[define(ES)] _scale_dtype: ElemType,
    #[define(EO)] _out_dtype: ElemType,
) {
    let config = comptime!(register_block(&bp, &problem));
    let w = w.tile_packed::<EC>(comptime!(space.clone()));
    let x = x.tile(comptime!(space.clone()));
    let mut scale_tiles = Sequence::new();
    #[unroll]
    for k in 0..scales.len() {
        scale_tiles.push(scales.index(k).tile(comptime!(space.clone())));
    }
    let out = out.tile(comptime!(space.clone()));
    // Each lane zeroes the window it owns: the output folds every step into what it holds.
    for cube in out.over(&bp.cubes()) {
        let out_cube = out.at(&cube);
        for plane in cube.over(&bp.planes()) {
            let out_plane = out_cube.at(&plane);
            for lane in plane.over(&bp.lanes(&problem)) {
                let mut out_lane = out_plane.at(&lane);
                out_lane.zero();
            }
        }
    }

    for cube in space.over(&bp.cubes()) {
        let out_cube = out.at(&cube);
        let w_cube = w.at(&cube);
        let x_cube = x.at(&cube);
        let scales_cube = at_all(&scale_tiles, &cube);
        for plane in cube.over(&bp.planes()) {
            let out_plane = out_cube.at(&plane);
            let w_plane = w_cube.at(&plane);
            let x_plane = x_cube.at(&plane);
            let scales_plane = at_all(&scales_cube, &plane);
            // The lane's share of the blocks, one stored word a step.
            for lane in plane.over(&bp.lanes(&problem)) {
                let mut out_lane = out_plane.at(&lane);
                let scales_lane = at_all(&scales_plane, &lane);
                out_lane.mma_scaled_with(
                    &w_plane.at(&lane),
                    &x_plane.at(&lane),
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
