//! The cmma boundary hoist: a memory accumulator bound for the [`Leaf::Cmma`] instruction
//! becomes an instance-resident fragment partition *before* the level walk — initialized
//! from this instance's windows, accumulated across the whole contraction, drained back
//! after. Fragments cannot be indexed at runtime, so residency is handled once here (and
//! the partition's own walk is the comptime microkernel), not per schedule.

use cubecl::{cmma::MatrixIdent, prelude::*, std::tensor::layout::CoordsDyn};

use crate::*;

/// Hoist `out` (a memory tile) into a resident [`CmmaPartition`], each fragment
/// initialized from its final window so the contraction accumulates onto the delivered
/// values. The partition carries `out`'s space, so the schedules walk it exactly like the
/// tile it replaces; [`Tile::at`] passes it through unchanged.
#[cube]
pub(crate) fn cmma_acc<Acc: Numeric, Lhs: Numeric>(
    out: &mut Tile<Acc>,
    lhs: &Tile<Lhs>,
) -> Tile<Acc> {
    let space = comptime!(out.space.clone());
    let (m_tiles, n_tiles) = comptime!(partition_shape(&space));
    let fin = comptime!(space.final_space());
    let m = comptime!(fin.extent_at(fin.rank() - 2));
    let n = comptime!(fin.extent_at(fin.rank() - 1));
    let k = comptime!(contracted_extent(&lhs.space, &space));

    let mut frags = Sequence::<CmmaData<Acc>>::new();
    #[unroll]
    for mi in 0..m_tiles {
        #[unroll]
        for ni in 0..n_tiles {
            let mut frag = CmmaData::<Acc>::alloc(MatrixIdent::Accumulator, m, n, k);
            let window = fragment_window(out, mi, ni);
            match &window.tile_kind {
                TileKind::Gmem(g) | TileKind::Smem(g) => frag.load_window(g),
                TileKind::Cmma(_) | TileKind::CmmaPartition(_) | TileKind::TmaGmem(_) => {
                    panic!("cmma_acc: the accumulator source must be memory")
                }
            }
            frags.push(frag);
        }
    }
    Tile::<Acc> {
        tile_kind: TileKind::new_CmmaPartition(CmmaPartition::<Acc> {
            frags,
            m_tiles,
            n_tiles,
        }),
        space: comptime!(space),
    }
}

/// Drain the resident partition back into this instance's final windows of `out`.
#[cube]
pub(crate) fn cmma_drain<Acc: Numeric>(out: &mut Tile<Acc>, acc: &Tile<Acc>) {
    match &acc.tile_kind {
        TileKind::CmmaPartition(p) => {
            #[unroll]
            for mi in 0..comptime!(p.m_tiles) {
                #[unroll]
                for ni in 0..comptime!(p.n_tiles) {
                    let mut window = fragment_window(out, mi, ni);
                    let frag = p.at(mi, ni);
                    match &mut window.tile_kind {
                        TileKind::Gmem(g) | TileKind::Smem(g) => frag.store_window(g),
                        TileKind::Cmma(_) | TileKind::CmmaPartition(_) | TileKind::TmaGmem(_) => {
                            panic!("cmma_drain: the accumulator sink must be memory")
                        }
                    }
                }
            }
        }
        TileKind::Gmem(_) | TileKind::Smem(_) | TileKind::Cmma(_) | TileKind::TmaGmem(_) => {
            panic!("cmma_drain: not a fragment-partition accumulator")
        }
    }
}

/// Descend to the `(mi, ni)` fragment's final window: an instance level hands this
/// instance a single region (`region(0)`, hardware position folded in); the partition
/// level takes the region at the comptime partition coordinates.
#[cube]
fn fragment_window<T: Numeric>(
    tile: &mut Tile<T>,
    #[comptime] mi: usize,
    #[comptime] ni: usize,
) -> Tile<T> {
    let space = comptime!(tile.space.clone());
    let mut sub = match comptime!(partition_level(&space)) {
        None => {
            let walk = Walk::over(tile.runtime_space());
            tile.at(&walk.region(0))
        }
        Some(_) => {
            let rows = comptime!(space.axis_at(space.rank() - 2));
            let cols = comptime!(space.axis_at(space.rank() - 1));
            tile.at(&region_at(
                comptime!(space.clone()),
                rows,
                mi,
                cols,
                ni.runtime(),
            ))
        }
    };
    match comptime!(sub.space.partitioner()) {
        Partitioner::Final(_) => sub,
        Partitioner::Level(_) => fragment_window(&mut sub, mi, ni),
    }
}

/// A [`Region`] of `space` at explicit coordinates: the comptime `frag` axis (a fragment
/// index must be comptime), the runtime `walk` axis (a contraction step may be a plain
/// loop variable), every other axis `0`. How the comptime microkernel reuses the runtime
/// windowing ([`Tile::at`]) without a runtime walk.
#[cube]
pub(crate) fn region_at(
    #[comptime] space: Space,
    #[comptime] frag_axis: Axis,
    #[comptime] frag_idx: usize,
    #[comptime] walk_axis: Axis,
    walk_idx: usize,
) -> Region {
    let mut coords = CoordsDyn::new();
    #[unroll]
    for p in 0..comptime!(space.rank()) {
        let axis = comptime!(space.axis_at(p));
        if comptime!(axis == frag_axis) {
            coords.push(frag_idx as u32);
        } else if comptime!(axis == walk_axis) {
            coords.push(walk_idx as u32);
        } else {
            coords.push(0u32);
        }
    }
    Region::new(coords, space)
}

/// The axis of `operand` the output drops — the contraction axis.
pub(crate) fn contracted_axis(operand: &Space, out: &Space) -> Axis {
    let contracted = operand.contracting(out);
    assert!(
        contracted.len() == 1,
        "cmma accumulator: the leaf contracts exactly one axis"
    );
    contracted[0]
}

/// The contraction depth `k`: the final-space extent of the contracted axis.
fn contracted_extent(operand: &Space, out: &Space) -> usize {
    operand.final_space().extent(contracted_axis(operand, out))
}

/// The per-instance tile count of `axis` at this level, `None` when it is runtime.
fn per_instance_tiles(level: &Space, axis: Axis) -> Option<usize> {
    let edge = level.partitioner().edge(axis);
    match level.partitioner().distribution(axis) {
        Distribution::Sequential => match level.extent_raw(axis) {
            Extent::Static(e) => Some(e.div_ceil(edge)),
            Extent::Dynamic => None,
        },
        Distribution::Spatial { coverage, .. } => match coverage {
            Coverage::TilesEach(t) => Some(t),
            Coverage::Instances(n) => match level.extent_raw(axis) {
                Extent::Static(e) => Some(e.div_ceil(edge).div_ceil(n)),
                Extent::Dynamic => None,
            },
        },
    }
}

/// Classify the current level of an accumulator's space: `None` for an *instance* level
/// (every axis hands this instance one tile — the walk visits a single region), or the
/// trailing-two-axes tile counts for the *partition* level (comptime sequential tiles
/// this instance owns wholesale). Anything else — a runtime count, a spatial axis mixed
/// with a multi-tile one, a multi-tile leading axis — cannot back resident fragments and
/// panics at comptime.
pub(crate) fn partition_level(space: &Space) -> Option<(usize, usize)> {
    if space.is_final() {
        return None;
    }
    if space
        .axes()
        .all(|a| per_instance_tiles(space, a) == Some(1))
    {
        return None;
    }
    let rank = space.rank();
    for (p, axis) in space.axes().enumerate() {
        assert!(
            matches!(
                space.partitioner().distribution(axis),
                Distribution::Sequential
            ),
            "cmma partition level: every axis below the instance split must be sequential"
        );
        let tiles = per_instance_tiles(space, axis)
            .expect("cmma partition level: tile counts must be comptime");
        assert!(
            p >= rank - 2 || tiles == 1,
            "cmma partition level: leading (batch) axes must hand out one tile"
        );
    }
    Some((
        per_instance_tiles(space, space.axis_at(rank - 2)).unwrap(),
        per_instance_tiles(space, space.axis_at(rank - 1)).unwrap(),
    ))
}

/// The whole remaining walk's fragment grid for one instance: `(1, 1)` when every level
/// is an instance level, else the single partition level's tile counts.
fn partition_shape(space: &Space) -> (usize, usize) {
    let mut shape = (1usize, 1usize);
    let mut level = space.clone();
    while !level.is_final() {
        if let Some(counts) = partition_level(&level) {
            assert!(
                shape == (1, 1),
                "cmma accumulator: at most one partition level"
            );
            shape = counts;
        }
        level = level.divide();
    }
    shape
}
