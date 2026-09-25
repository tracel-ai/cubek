//! How one axis of a walk's space is distributed at its level ([`AxisDistribution`]), settled on the host so
//! the walk asks each distribution for its share and its position instead of asking the level six
//! questions per axis, and where an instance finds its own position ([`ComputeScope::position`],
//! [`CubeAxis::position`]).

use cubecl::prelude::*;

use crate::{
    Axis, ComputeScope, Count, CubeAxis, Extent, Integer, IntegerExpand, Level, Space, Spread,
};

/// One axis of a level, as a walk distributes it: walked whole by every instance, or distributed
/// over the level's scope.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) enum AxisDistribution {
    /// Every instance steps the whole grid along this axis.
    Walked,
    /// The grid is distributed over the scope's instances ([`Distribution`]).
    Distributed(Distribution),
}

/// How an axis's grid is distributed over a level's scope: one tile an instance, or `across` of them in
/// runs.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct Distribution {
    /// The scope whose instances take the tiles.
    pub(crate) scope: ComputeScope,
    /// The grid dimension a cube level's axis rides.
    pub(crate) dim: Option<CubeAxis>,
    pub(crate) spread: Spread,
    /// The workers an [`AllAcross`](Count::AllAcross) axis is distributed to in runs; `None`
    /// where each worker takes one tile.
    pub(crate) across: Option<usize>,
    /// Whether the tiles are taken in turns by as many units as the launch runs
    /// ([`Count::Distributed`]): the instances are the launch's, not the grid's.
    pub(crate) in_turns: bool,
    /// Whether the host proved every worker's run whole, so the kernel skips clamping it.
    pub(crate) divides: bool,
    /// The later axes sharing this axis's hardware dimension, whose instance counts weight
    /// this axis's digit (the earlier axis is the more significant digit).
    pub(crate) inner: Vec<usize>,
    /// The instance weight of the same-dimension axes inside this one that the walked space
    /// does not span, read off the level since the space has dropped them.
    pub(crate) unspanned: usize,
    /// Which of the two jointly decoded in-plane positions this axis takes, when the cube
    /// level distributes its boxes in a swizzled order.
    pub(crate) swizzled: Option<usize>,
}

impl AxisDistribution {
    /// How `level` distributes axis `p` of `space`, `swizzled` the two in-plane positions where the
    /// cube level states an order.
    pub(crate) fn new(
        level: &Level,
        space: &Space,
        p: usize,
        swizzled: Option<(usize, usize)>,
    ) -> AxisDistribution {
        let axis = space.axis_at(p);
        if !level.distributes(axis) {
            return AxisDistribution::Walked;
        }
        let dim = level.cube_axis(axis);
        let same_dim = |q: usize| {
            let other = space.axis_at(q);
            level.distributes(other) && level.cube_axis(other) == dim
        };
        AxisDistribution::Distributed(Distribution {
            scope: level
                .coverage()
                .scope()
                .expect("a level that distributes an axis distributes it over a scope"),
            dim,
            spread: level.cut(axis).spread,
            across: match level.cut(axis).count {
                Count::AllAcross(workers) => Some(workers),
                Count::Stated(_) | Count::All | Count::Distributed(_) => None,
            },
            in_turns: matches!(level.cut(axis).count, Count::Distributed(_)),
            divides: AxisDistribution::divides(level, space, axis),
            inner: ((p + 1)..space.rank()).filter(|&q| same_dim(q)).collect(),
            unspanned: AxisDistribution::unspanned_weight(level, space, axis),
            swizzled: match swizzled {
                Some((x, _)) if x == p => Some(0),
                Some((_, y)) if y == p => Some(1),
                _ => None,
            },
        })
    }

    /// Whether every worker's run along an axis distributed across workers is the full one: the grid
    /// divides the worker count, provable only of a static extent. Any other count distributes one tile
    /// a worker, which every grid divides.
    fn divides(level: &Level, space: &Space, axis: Axis) -> bool {
        match level.cut(axis).count {
            Count::AllAcross(workers) => match space.extent_raw(axis) {
                Extent::Static(extent) => extent
                    .div_ceil(level.cut(axis).tile)
                    .is_multiple_of(workers),
                Extent::Dynamic => false,
            },
            // The units are the launch's, so nothing here can prove they divide the count.
            Count::Distributed(_) => false,
            Count::Stated(_) | Count::All => true,
        }
    }

    /// The instance-index weight `space`'s own axis list cannot see: the instance counts of the
    /// same-dimension axes *inside* `axis` that `level` distributes and `space` does not span. The
    /// odometer is the level's, so an operand divides out contracted axes it does not span.
    ///
    /// Reading omitted axes as weight `1` aliases the outer digits onto one value, so this panics
    /// where such an axis has no comptime count: assuming `1` would be exactly that aliasing.
    pub(crate) fn unspanned_weight(level: &Level, space: &Space, axis: Axis) -> usize {
        let dim = level.cube_axis(axis);
        level
            .axes()
            .iter()
            .skip_while(|&&a| a != axis)
            .skip(1)
            .filter(|&&a| !space.contains(a) && level.distributes(a) && level.cube_axis(a) == dim)
            .map(|&a| {
                level.cut(a).count.stated().unwrap_or_else(|| {
                    panic!(
                        "AxisDistribution: {a:?} is distributed inside {axis:?} over the same scope but this \
                         operand does not span it, and its instance count is not comptime, so \
                         {axis:?}'s digit of the instance index cannot be decoded"
                    )
                })
            })
            .product()
    }
}

#[cube]
impl AxisDistribution {
    /// The tiles the instance at `pos` of `instances` takes of a `grid` distributed in runs of `run`:
    /// the whole run where the host proved the grid `divides`, else the run cut where the grid
    /// ends (contiguous) or the turns left to it (interleaved). Saturating: an instance past the
    /// grid takes nothing.
    pub(crate) fn tiles(
        grid: usize,
        pos: usize,
        instances: usize,
        run: usize,
        #[comptime] spread: Spread,
        #[comptime] divides: bool,
    ) -> usize {
        if divides {
            run
        } else {
            match spread {
                Spread::Contiguous => {
                    let start = pos.times(run);
                    run.min_with(grid.max_with(start).minus(start))
                }
                Spread::Interleaved => grid
                    .max_with(pos)
                    .minus(pos)
                    .plus(instances.minus(1usize))
                    .divided_by(instances),
            }
        }
    }

    /// The raw hardware position of the instances this distribution hands tiles to, before it is folded
    /// through the axis's shared-dimension stride.
    pub(crate) fn hardware(
        #[comptime] compute_scope: ComputeScope,
        #[comptime] dim: Option<CubeAxis>,
    ) -> usize {
        match comptime!(compute_scope) {
            ComputeScope::Cube => CubeAxis::position(comptime!(
                dim.expect("a cube level's distributed axis rides a grid dimension")
            )),
            ComputeScope::Plane | ComputeScope::Unit => ComputeScope::position(compute_scope),
        }
    }
}

#[cube]
impl ComputeScope {
    /// This instance's position within `compute_scope`: which plane of the cube, or which unit of the
    /// plane. A cube's is per grid dimension ([`CubeAxis::position`]).
    ///
    /// `cube_dim = new_2d(plane_size, num_planes)`: `Y` is the plane index, `X` the plane-relative
    /// unit. Units agree on `UNIT_POS_Y`, so they cooperate. The plane-relative unit, not the flat
    /// `UNIT_POS`: flat would fold in `UNIT_POS_Y` and double-count a sibling plane axis's digit.
    pub fn position(#[comptime] compute_scope: ComputeScope) -> usize {
        match comptime!(compute_scope) {
            ComputeScope::Plane => UNIT_POS_Y as usize,
            ComputeScope::Unit => UNIT_POS_X as usize,
            ComputeScope::Cube => {
                panic!(
                    "ComputeScope::position: a cube has one position per grid dimension; say which"
                )
            }
        }
    }
}

#[cube]
impl CubeAxis {
    /// This cube's position on grid dimension `dim`.
    pub fn position(#[comptime] dim: CubeAxis) -> usize {
        let cube_pos = match comptime!(dim) {
            CubeAxis::X => CUBE_POS_X,
            CubeAxis::Y => CUBE_POS_Y,
            CubeAxis::Z => CUBE_POS_Z,
        };
        cube_pos as usize
    }
}
