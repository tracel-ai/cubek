//! [`Resident`]: the accumulator's register-tier decorator, [`Staging`](crate::Staging)'s
//! write-side dual. A memory accumulator bound for the [`Leaf::Cmma`] instruction runs its
//! whole contraction on a register-resident fragment partition: initialized from the
//! accumulator's windows once, accumulated across the whole walk, written back once (the
//! classic global matmul's `init_accumulator` / epilogue). Where `Staging` refills per
//! region, residency brackets the whole contraction — fragments cannot be indexed at
//! runtime, so it is entered once at the outermost cmma boundary, not per schedule.
//!
//! `contract` is a hand-written expand method for the same reason as `Staging`'s
//! `fill`/`consume`: the write-back must follow the caller-defined body, a `Drop` guard
//! can't emit ops in cubecl, and `#[cube]` rejects `impl Trait` args.

use cubecl::{cmma::MatrixIdent, prelude::*, unexpanded};

use crate::*;

/// The register-resident form of a memory accumulator: the fragment partition standing in
/// for the tile it decorates. Built by [`new`](Resident::new), consumed by
/// [`contract`](Resident::contract).
#[derive(CubeType)]
pub struct Resident<T: Numeric> {
    acc: Tile<T>,
}

/// Enter register residency for `out` and run its contraction there: the recursion
/// continues on the fragment partition in place of the memory tile (it carries the same
/// space, so the schedules walk it exactly like the tile it replaces), and the write-back
/// follows inside `contract`.
#[cube]
pub(crate) fn mma_resident<Acc: Numeric, Lhs: Numeric, Rhs: Numeric>(
    out: &mut Tile<Acc>,
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
) {
    let mut acc = Resident::new(out, lhs);
    acc.contract(out, |frags| frags.mma(lhs, rhs));
}

#[cube]
impl<Acc: Numeric> Resident<Acc> {
    /// Promote `out` to a resident fragment partition, each fragment initialized from its
    /// final window so the contraction accumulates onto the delivered values.
    pub fn new<Lhs: Numeric>(out: &mut Tile<Acc>, lhs: &Tile<Lhs>) -> Resident<Acc> {
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
                        panic!("Resident: the source must be memory")
                    }
                }
                frags.push(frag);
            }
        }
        Resident::<Acc> {
            acc: Tile::<Acc> {
                tile_kind: TileKind::new_CmmaPartition(CmmaPartition::<Acc> {
                    frags,
                    m_tiles,
                    n_tiles,
                }),
                space: comptime!(space),
            },
        }
    }

    /// Drain the resident partition back into the final windows of the tile it decorates.
    fn writeback(&self, out: &mut Tile<Acc>) {
        match &self.acc.tile_kind {
            TileKind::CmmaPartition(p) =>
            {
                #[unroll]
                for mi in 0..comptime!(p.m_tiles) {
                    #[unroll]
                    for ni in 0..comptime!(p.n_tiles) {
                        let mut window = fragment_window(out, mi, ni);
                        let frag = p.at(mi, ni);
                        match &mut window.tile_kind {
                            TileKind::Gmem(g) | TileKind::Smem(g) => frag.store_window(g),
                            TileKind::Cmma(_)
                            | TileKind::CmmaPartition(_)
                            | TileKind::TmaGmem(_) => {
                                panic!("Resident: the sink must be memory")
                            }
                        }
                    }
                }
            }
            TileKind::Gmem(_) | TileKind::Smem(_) | TileKind::Cmma(_) | TileKind::TmaGmem(_) => {
                panic!("Resident: write-back expects the fragment partition")
            }
        }
    }
}

impl<Acc: Numeric> Resident<Acc> {
    /// Run `compute` on the resident accumulator, then write it back to `out` — the
    /// closure-scoped lifecycle, mirroring [`Staging`](crate::Staging)'s `fill`/`consume`.
    /// See [`ResidentExpand::__expand_contract_method`].
    pub fn contract(&mut self, _out: &mut Tile<Acc>, _compute: impl FnOnce(&mut Tile<Acc>)) {
        unexpanded!()
    }
}

impl<Acc: Numeric> ResidentExpand<Acc> {
    pub fn __expand_contract_method<F>(
        &mut self,
        scope: &Scope,
        out: &mut TileExpand<Acc>,
        compute: F,
    ) where
        F: FnOnce(&Scope, &mut TileExpand<Acc>),
    {
        compute(scope, &mut self.acc);
        self.__expand_writeback_method(scope, out);
    }
}

/// Descend to the `(mi, ni)` fragment's final window: an instance level hands this
/// instance a single region (`region(0)`, hardware position folded in); the partition
/// level takes the comptime region at the partition coordinates.
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
        Some(_) => tile.at_comptime(comptime!(CRegion::trailing(&space, mi, ni))),
    };
    match comptime!(sub.space.partitioner()) {
        Partitioner::Final(_) => sub,
        Partitioner::Level(_) => fragment_window(&mut sub, mi, ni),
    }
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

/// Classify the current level of an accumulator's space: `None` for an *instance* level (one
/// tile per axis), or the trailing-two-axes tile counts for the *partition* level (comptime
/// sequential tiles this instance owns wholesale). Anything else cannot back resident
/// fragments and panics at comptime.
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
