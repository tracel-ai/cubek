//! The quantized decode gemv kernel: the space it runs over and the walk written out.

use cubecl::prelude::*;
use cubek_tile::{
    Level, Levels, Partitioning, RegisterBlock, Semiring, Space, TileArg, scale_tile,
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

/// The routine's levels, stated **from the leaf up, in counts**: one lane's turn, its rows
/// against every block one word of scales covers, whole; the lanes of a plane, taking those
/// turns between them; the walk over the blocks past the plane's turn; the planes of a cube; a
/// cube per strip.
///
/// The walk is the level the old statement did not have: its lanes level both dealt the
/// contraction to the lanes *and* was the walk the plane made over what was left of it. Leaf up
/// a level says only how many of the thing below, so the walk is its own level, above the lanes.
/// Below them there is none: a lane's turn is one tile, and stepping it in single blocks would
/// read a word of scales in halves.
pub fn quant_gemv_levels(bp: &QuantGemvBlueprint, problem: &QuantGemvProblem) -> Vec<Level> {
    Levels::leaf(&[
        (M, bp.rows_per_lane),
        (KB, problem.scales_per_word()),
        (KI, problem.block),
    ])
    .lanes(&[(M, bp.groups()), (KB, bp.block_lanes)])
    .interleaved(KB)
    .walk_every(&[KB])
    .planes(&[(M, bp.rows_per_cube / bp.rows_per_plane)])
    .cubes(&[M])
    .build()
}

impl QuantGemvBlueprint {
    /// The space with the levels that cut it: what the leaf and the overhangs are read off.
    pub fn partitioning(&self, problem: &QuantGemvProblem) -> Partitioning {
        Partitioning::new(quant_gemv_space(problem), quant_gemv_levels(self, problem))
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

    /// The fold's level: `rows_per_lane` rows per aligned lane group, the group's lanes taking
    /// turns at the contraction, a word of scales each. What the zeroing and the drain name
    /// beside their loops, read off the list rather than stated twice.
    pub fn lanes(&self, problem: &QuantGemvProblem) -> Level {
        quant_gemv_levels(self, problem)[3].clone()
    }
}

/// The register block the leaf runs under: one scalar accumulator per row a lane owns, per
/// value of the word it takes a step.
pub fn register_block(bp: &QuantGemvBlueprint, problem: &QuantGemvProblem) -> RegisterBlock {
    RegisterBlock::new(bp.rows_per_lane * problem.factor())
}

/// `y = (W ⊗ s) · x`.
///
/// The weight arrives as `u32` words and unpacks at the read ([`TileArg::tile_as`]); each
/// scale level arrives as its own tensor and folds in at the contraction
/// ([`cubek_tile::Scaled::scaled`]), written on the weight. A scheme always has the block
/// level and may have the factor over the whole tensor; unbound, that one scales by one. Nothing
/// here mentions a quantization scheme, a block size or a scale binding riding the weight: which
/// values one scale covers is the scales operand's own axes, stated in the space.
///
/// Every operand keeps its own element: `EC` is what the words decode to, `EX` what the
/// activation buffer holds, `ES` the scales', `EO` the output's, and the leaf casts each into
/// the accumulator as it always does. The activation is served in `VX`-wide lines along the
/// contraction, one stored word's worth a step, and the output scalar, because each lane holds
/// a partial of its group's cell.
///
/// Four levels, each one region per instance: the cube's strip of rows, the plane's group of
/// rows, the plane's walk over the blocks, and the lane's rows against its turn at them, which
/// the leaf folds across the plane's lanes as it writes.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn quant_gemv_kernel<EC: Numeric, EX: Numeric, ES: Numeric, EO: Numeric, VX: Size, VO: Size>(
    w: &TileArg<'_, u32, Const<1>>,
    x: &TileArg<'_, EX, VX>,
    block_scale: &TileArg<'_, u32, Const<1>>,
    global_scale: &ComptimeOption<TileArg<'static, u32, Const<1>>>,
    out: &TileArg<'_, EO, VO>,
    space: Partitioning,
    #[comptime] bp: QuantGemvBlueprint,
    #[comptime] problem: QuantGemvProblem,
    #[define(EC)] _served_dtype: ElemType,
    #[define(EX)] _x_dtype: ElemType,
    #[define(ES)] _scale_dtype: ElemType,
    #[define(EO)] _out_dtype: ElemType,
) {
    let config = comptime!(register_block(&bp, &problem));
    let w = w
        .tile_as::<EC>(comptime!(space.clone()))
        .scaled(&ComptimeOption::new_Some(
            block_scale.tile_as::<ES>(comptime!(space.clone())),
        ))
        .scaled(&scale_tile::<ES>(global_scale, comptime!(space.clone())));
    let x = x.tile(comptime!(space.clone()));
    let out = out.tile(comptime!(space.clone()));
    // Each lane zeroes the window it owns: the output folds every step into what it holds.
    // The output spans no contraction, so the lanes level is named rather than descended to.
    let lanes = comptime!(bp.lanes(&problem));
    for cube in &out {
        let out_cube = out.at(&cube);
        for plane in cube {
            let out_plane = out_cube.at(&plane);
            for lane in out_plane.over(&comptime!(lanes.clone())) {
                let mut out_lane = out_plane.at(&lane);
                out_lane.zero();
            }
        }
    }

    for cube in space {
        let out_cube = out.at(&cube);
        let w_cube = w.at(&cube);
        let x_cube = x.at(&cube);
        for plane in cube {
            let out_plane = out_cube.at(&plane);
            let w_plane = w_cube.at(&plane);
            let x_plane = x_cube.at(&plane);
            // The plane's walk over the turns its lanes take, then the lane's own, the blocks
            // its word of scales covers.
            for turn in plane {
                for lane in turn {
                    let mut out_lane = out_plane.at(&lane);
                    out_lane.mma_scaled_with(
                        &w_plane.at(&lane),
                        &x_plane.at(&lane).plain(),
                        config,
                        Semiring::SUM_PROD,
                    );
                }
            }
        }
    }
}
