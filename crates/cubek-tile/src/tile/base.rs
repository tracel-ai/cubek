//! The [`Tile`]: one operand's data as a [`TileKind`] backing store, plus the comptime
//! [`Space`] it projects. Structure only; each store's own data and leaves live in its file
//! ([`mem`](crate::MemData), [`cmma`](crate::CmmaData), [`tma`](crate::TmaData)).

use cubecl::{prelude::*, std::tensor::layout::CoordsDyn, unexpanded};

use cubecl::zspace::SmallVec;

use crate::*;

/// A tile's backing store. Every variant is lifetime-free (a `Box<[T]>` or a
/// [`cmma::Matrix`](cubecl::cmma::Matrix)); [`view`](Tile::view) rebuilds a borrowed view on
/// demand. `Clone` copies the handle, not the cells, which is how later ring slots reuse a fixed
/// operand's first buffer and is only sound where nothing rewrites the buffer afterwards.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum TileKind<T: Numeric> {
    Gmem(MemData<T>),
    Smem(MemData<T>),
    /// One plane-level tile: owned by a plane and sliced across its lanes, so never addressable
    /// (no memory view). The [`Instruction`] picks the encoding; the contraction is its own.
    PlaneTile(PlaneTile<T>),
    /// The grid of plane tiles one plane owns, `m_tiles × n_tiles`, comptime-indexed; only a
    /// static walk's regions (constant coordinates) can select through it.
    PlanePartition(PlanePartition<T>),
    /// A TMA tensor-map source: not element-addressable, its only sink is a hardware bulk
    /// copy into shared memory. Launched via [`TmaTileArg`](crate::TmaTileArg).
    TmaGmem(TmaData<T>),
    /// A read-only source evaluated from logical coordinates with no backing buffer.
    Procedural(ProceduralData<T>),
}

#[cube]
impl<T: Numeric> TileKind<T> {
    /// Whether a level must be walked with comptime coordinates. Fragments can't be picked out by
    /// a runtime region, so a plane partition sitting at a real partition level forces the unrolled
    /// walk. Comptime.
    pub(crate) fn static_level(&self, #[comptime] space: Space) -> comptime_type!(bool) {
        match self {
            TileKind::PlanePartition(_) => comptime!(matches!(
                space.partitioner(),
                Partitioner::Level(l) if matches!(l.role(), LevelRole::Partition)
            )),
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(false)
            }
        }
    }

    /// Whether a staged walk here must unroll for correctness. When the level cuts a fragment
    /// partition, each region picks its own block of fragments, which needs comptime coordinates.
    /// A 1×1 level (a k-step walk) cuts nothing and passes the partition through, so it stays a
    /// plain runtime loop. Comptime.
    pub(crate) fn cuts_partition(&self, #[comptime] space: Space) -> comptime_type!(bool) {
        match self {
            TileKind::PlanePartition(_) => comptime!(space.cuts_tiles()),
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(false)
            }
        }
    }
}
/// One operand's data: a runtime [`TileKind`] backing store and the comptime [`Space`] it
/// projects. `T` is the element the tile serves and computes in; its physical vector width is a
/// storage detail inside the [`TileKind`], read back with [`vector_size`](Tile::vector_size).
/// What an operand is at the instruction is the operand's own statement (the finest
/// [`Residence::Register`] stage), so operands that disagree meet the kind-pairing panics there.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Tile<T: Numeric> {
    pub tile_kind: TileKind<T>,
    #[cube(comptime)]
    pub space: Space,
}

/// The one physical dim whose bound is `axis`'s own extent: it carries `axis` alone, at
/// coefficient `1`. `None` otherwise, which is either of the two ways a bound stops being that
/// extent: a gather, where the dim holds the receptive field several axes reach over, and storage
/// tiling, where the extent is the product over the dims the axis is split across.
fn bound_states(projection: &Projection, axis: Axis) -> Option<usize> {
    // A broadcast axis has no dim to read a bound off: the operand is constant along it, so its
    // buffer holds nothing that sizes it.
    if !projection.addresses(axis) {
        return None;
    }
    match projection.carriers(axis)[..] {
        [pa] if projection.physical_axis(pa).is_identity(axis) => Some(pa),
        _ => None,
    }
}

/// The physical dim in this tile's window bounds that `axis`'s runtime extent is read off. A
/// direct operand maps each axis 1:1; anything else has to be answered by an operand of the same
/// operation that does ([`Tile::witnesses`]).
fn bound_position(projection: &Projection, axis: Axis) -> usize {
    bound_states(projection, axis).unwrap_or_else(|| {
        panic!(
            "Tile::runtime_extent: no bound of this operand is {axis:?}'s own extent (it gathers \
             over it, or splits it across storage fragments); ask an operand that witnesses it"
        )
    })
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// Whether the instruction can consume this operand in its current physical form, read off
    /// the register stages still ahead of it. Opaque fragment transports require shared memory
    /// or an already materialized fragment; scalar and manual readers can address their source
    /// directly.
    fn reads_in_place(&self) -> comptime_type!(bool) {
        match &self.tile_kind {
            TileKind::Smem(_) | TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                comptime!(true)
            }
            TileKind::Gmem(_) | TileKind::TmaGmem(_) | TileKind::Procedural(_) => {
                let plan = self.stage_plan();
                let staged = comptime!(plan.stages_to_registers());
                comptime!(match self.space.instruction().filter(|_| staged) {
                    None | Some(Instruction::Registers { .. }) => true,
                    Some(Instruction::Mma { io }) => {
                        matches!(io.lhs_load_method, LoadMethod::Manual)
                            && matches!(io.rhs_load_method, LoadMethod::Manual)
                    }
                    Some(Instruction::Cmma) => false,
                })
            }
        }
    }

    /// Evaluate a procedural tile at scalar logical coordinates relative to its current region.
    pub fn procedural_value(&self, pos: Coords<u32>) -> T {
        match &self.tile_kind {
            TileKind::Procedural(data) => data.evaluate(&pos, comptime!(self.space.clone())),
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_) => panic!("Tile::procedural_value: tile is not procedural"),
        }
    }

    /// How this operand's bytes move: a strided cooperative copy or a TMA hardware bulk
    /// copy. Comptime (the kind is fixed at trace); drives the staging sync. A resident
    /// fragment has no bytes to move, so go through
    /// [`stage_source`](Tile::stage_source) rather than calling this on one.
    pub fn delivery(&self) -> comptime_type!(Delivery) {
        match &self.tile_kind {
            TileKind::Gmem(_) | TileKind::Smem(_) => comptime!(Delivery::Copy),
            TileKind::TmaGmem(_) => comptime!(Delivery::Tma),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::delivery: a resident fragment is not a stage source")
            }
            // A procedural source is cooperatively materialized into its stage.
            TileKind::Procedural(_) => comptime!(Delivery::Procedural),
        }
    }

    /// How a staging slot obtains this operand: transport from a backing store, or fragments a
    /// level above already placed in registers. A fragment has no bytes any transport can move, so
    /// its slot holds the fragments themselves and each read selects the region's block out of them
    /// by comptime coordinate ([`AtRegion`](crate::SlotPayload::AtRegion)).
    pub(crate) fn stage_source(&self) -> comptime_type!(StageSource) {
        match &self.tile_kind {
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                comptime!(StageSource::ResidentFragment)
            }
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                let delivery = self.delivery();
                comptime!(StageSource::Transport(delivery))
            }
        }
    }

    /// The runtime map (dynamic coefficients and origin phase residues) of this tile.
    pub(crate) fn runtime_map(&self) -> RuntimeMap {
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => g.map.clone(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => RuntimeMap::integral(comptime!(self.space.rank())),
        }
    }

    /// Where this operand lives at each level from here down, and how a materialized level lays
    /// its buffer out. A resident fragment is never a fill source and has no levels left to state,
    /// so it answers the all-[`InPlace`](Residence::InPlace) default.
    pub fn stage_plan(&self) -> comptime_type!(StagePlan) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.stage_plan(),
            TileKind::TmaGmem(t) => comptime!(t.stage.clone()),
            TileKind::Procedural(p) => comptime!(p.stage.clone()),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                comptime!(StagePlan::in_place())
            }
        }
    }

    /// Where this operand lives at the level whose output space is `out`.
    /// `InPlace` is honoured as stated, so a level materializes an operand only when the operand
    /// asked it to; the one thing that cannot be honoured is a leaf reading a source it cannot
    /// address, which is checked at the level that feeds it.
    pub fn residence(&self, #[comptime] out: &Space) -> comptime_type!(Residence) {
        let plan = self.stage_plan();
        let requested = comptime!(plan.head());
        if comptime!(requested == Residence::InPlace) {
            let reads_in_place = self.reads_in_place();
            comptime!(if out.partitioner().next().is_final() && !reads_in_place {
                panic!(
                    "Tile::residence: a {:?} register stage cannot read this operand's current \
                     physical form in place; materialize it with Residence::Smem at \
                     some level above the instruction",
                    self.space.instruction()
                );
            });
            comptime!(Residence::InPlace)
        } else {
            let procedural = self.is_procedural();
            comptime!(if procedural && matches!(requested, Residence::Register) {
                panic!(
                    "Tile::residence: a procedural source has no plane-fragment transport; state \
                     Residence::Smem to materialize it into shared memory, or Residence::InPlace \
                     to evaluate it at the leaf"
                );
            });
            comptime!(requested)
        }
    }

    /// Physical vectorization of the backing store: the `Vector<T, vector_size>` line
    /// width the leaf reconstructs. A launched memory tile carries its operand's vector
    /// size; a cmma fragment and a tma source are scalar (`1`).
    pub fn vector_size(&self) -> comptime_type!(usize) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.store.vector_size,
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(1usize)
            }
        }
    }

    /// What this tile's cells are to the plane's lanes: whole, or a partial only true once
    /// combined across the plane. A resident form inherits it from the memory it was promoted
    /// from: the split is the space's, not the storage's.
    pub(crate) fn lane_share(&self) -> comptime_type!(LaneShare) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.lanes.share,
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(LaneShare::Whole)
            }
        }
    }

    /// What one instance holds of this tile's cells. A form that is not memory holds whole cells
    /// by construction: it is a plane's own registers, which no other instance writes.
    pub(crate) fn split_share(&self) -> comptime_type!(SplitShare) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => comptime!(d.split_share),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(SplitShare::Whole)
            }
        }
    }

    /// What a write to this tile does to the cell it lands on. A form that is not memory has no
    /// cell to land on, so it answers with the only thing a later store into memory can be asked
    /// to do, and the memory it drains into answers for itself.
    pub(crate) fn write(&self) -> comptime_type!(Write) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => comptime!(d.access.write),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(Write::Replace)
            }
        }
    }

    /// Ask this accumulator to start from `init_from`, and answer what actually took. Asking
    /// rather than deciding keeps a kind that cannot take the request from being listed anywhere
    /// else. A destination that folds always answers [`Identity`](InitFrom::Identity): its cells
    /// belong to several instances, so the buffer must already hold the fold's identity (the
    /// launch's obligation), and `Cell` would seed a cell the siblings are folding into.
    pub(crate) fn request_init_from(
        &mut self,
        #[comptime] init_from: InitFrom,
    ) -> comptime_type!(InitFrom) {
        match &mut self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => {
                let init_from = comptime!(match d.access.write {
                    Write::Accumulate => InitFrom::Identity,
                    Write::Replace => init_from,
                });
                d.set_init_from(comptime!(init_from));
                comptime!(init_from)
            }
            // A promoted fragment states its own init and never reads a cell back to begin with,
            // so the request does not take and its caller seeds instead.
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(InitFrom::Cell)
            }
        }
    }

    /// This operand's decode site ([`DequantAt`]). A tile with nothing to decode answers
    /// [`DequantAt::Load`]: served and stored are the same element, so its load already delivers
    /// what the read wants and the stage takes that element. A tma source is never quantized, so it
    /// answers the same for the same reason, not because a bulk copy could decode: it cannot.
    pub(crate) fn dequant_at(&self) -> comptime_type!(DequantAt) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.dequant_at(),
            TileKind::TmaGmem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::Procedural(_) => {
                comptime!(DequantAt::Load)
            }
        }
    }

    /// How this tile's values sit in memory; see [`MemData::packing`]. A resident fragment, a tma
    /// source and a procedural tile are never quantized.
    pub(crate) fn packing(&self) -> comptime_type!(Packing) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.packing(),
            TileKind::TmaGmem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::Procedural(_) => {
                comptime!(Packing::Plain)
            }
        }
    }

    /// Whether this tile's buffer is addressed through a mapping whose windows may *overlap*, so
    /// the only read surface describing it is the N-D one ([`nd`](Tile::nd)) and no window is
    /// dense. False for a [`direct`](Projection::direct) operand and equally for one whose axes
    /// [partition](Composition::Disjoint) a physical axis, a partition being a bijection.
    pub fn gathered(&self) -> comptime_type!(bool) {
        let projection = self.projection();
        comptime!(projection.composition() == Composition::Overlapping)
    }

    /// Whether this tile evaluates values from coordinates instead of a backing buffer.
    pub(crate) fn is_procedural(&self) -> comptime_type!(bool) {
        match &self.tile_kind {
            TileKind::Procedural(_) => comptime!(true),
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_) => comptime!(false),
        }
    }

    /// The factorization this tile's values state, if any: `Some(n)` for a recipe presenting `n`
    /// separable factors, `None` for a tile read from a buffer or a recipe that only answers as a
    /// whole. A rank-one factorization is `Some(1)`, which a consumer can still exploit, and so is
    /// deliberately not the same answer as `None`.
    pub(crate) fn factors(&self) -> comptime_type!(Option<usize>) {
        match &self.tile_kind {
            TileKind::Procedural(data) => data.factors(),
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_) => comptime!(None),
        }
    }

    /// This operand's window sign, for a stage recording where it was filled from.
    pub(crate) fn window_signed(&self) -> comptime_type!(bool) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => comptime!(d.window.signed),
            _ => comptime!(false),
        }
    }

    /// This operand's per-axis boundary handling, read by a stage filled from it: the stage's own
    /// list is empty (it never overhangs its buffer), so the padding policy has to come from here.
    pub(crate) fn window_boundaries(
        &self,
    ) -> comptime_type!(SmallVec<[Option<Boundary>; MAX_AXES]>) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => comptime!(d.window.boundaries.clone()),
            _ => comptime!(SmallVec::new()),
        }
    }

    /// The physical boundaries factor-local normalization tests. A shared-memory tile preserves
    /// these on its source window even though its resident window itself no longer overhangs.
    pub(crate) fn separable_boundaries(
        &self,
    ) -> comptime_type!(SmallVec<[Option<Boundary>; MAX_AXES]>) {
        match &self.tile_kind {
            TileKind::Gmem(d) => comptime!(d.window.boundaries.clone()),
            TileKind::Smem(d) =>
            {
                #[comptime]
                match &d.source_window {
                    ComptimeOption::Some(source) => comptime!(source.boundaries.clone()),
                    ComptimeOption::None => comptime!(SmallVec::new()),
                }
            }
            TileKind::Procedural(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_) => comptime!(SmallVec::new()),
        }
    }

    /// The separable factor-normalization request, if one was attached to this procedural tile.
    /// Backed tiles answer `None`, as they have no factor evaluation for the gather leaf to alter.
    pub(crate) fn factor_normalization(
        &self,
    ) -> comptime_type!(Option<(TapMask, DivGuard, Space)>) {
        match &self.tile_kind {
            TileKind::Procedural(data) => comptime!(data.normalization.clone()),
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_) => comptime!(None),
        }
    }

    /// One factor of a separable recipe, evaluated at `pos`. Only the coordinate along the axis
    /// that factor reads matters, which is what lets the contraction walk it in 1-D. Asking
    /// [`factors`](Tile::factors) first is the whole precondition.
    pub(crate) fn separable_factor(&self, pos: CoordsDyn, #[comptime] factor: usize) -> T {
        match &self.tile_kind {
            TileKind::Procedural(data) => {
                data.evaluate_factor_dyn(&pos, factor, comptime!(self.space.clone()))
            }
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_) => {
                panic!(
                    "Tile::separable_factor: a tile read from a buffer states no factorization, \
                     so `factors` answered `None` and there is no factor {factor} to evaluate"
                )
            }
        }
    }

    /// The axes of `walk` this tile is constant along, so one read serves every position of all
    /// of them at once. Two spellings, one fact: an axis the tile's space does not span, and an
    /// axis it spans but whose projection addresses nothing. A consumer asking how far out of a
    /// nest a read lifts wants both. The [gathered nest](crate::FactorReuse) asks the same of a
    /// procedural factor through its recipe, which reads its axes rather than addressing them.
    pub(crate) fn invariant_over(&self, #[comptime] walk: Space) -> comptime_type!(Vec<Axis>) {
        let projection = self.projection();
        comptime!(
            walk.axes()
                .filter(|axis| !self.space.contains(*axis) || !projection.addresses(*axis))
                .collect::<Vec<_>>()
        )
    }

    /// Whether this tile can state `axis`'s runtime size for its operation: it spans the axis
    /// [`Dynamic`](crate::Extent), has a buffer to read a bound off, and that bound is the axis's
    /// own extent ([`bound_states`]). Spanning an axis and being able to supply it are separate
    /// questions, so an operation sizes each `Dynamic` axis from whichever operand witnesses it
    /// and lets the others pass ([`witnessed_space`]).
    pub fn witnesses(&self, #[comptime] axis: Axis) -> comptime_type!(bool) {
        let bounded = self.bounded();
        let projection = self.projection();
        comptime!(
            bounded
                && self.space.contains(axis)
                && self.space.is_dynamic(axis)
                && bound_states(&projection, axis).is_some()
        )
    }

    /// How this tile's logical axes address its buffer's physical ones. A fragment and a tma source
    /// have no buffer to project onto, so they answer [`direct`](Projection::direct) over their own
    /// space, which is what every non-gather operand carries anyway.
    pub(crate) fn projection(&self) -> comptime_type!(Projection) {
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => comptime!(g.projection.clone()),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(Projection::direct_over(&self.space))
            }
        }
    }

    /// Window this tile down to `region`, no copy. Each tile projects `region` onto its own axes, so
    /// `lhs ∈ {M,K}` and `out ∈ {M,N}` line up on their own; the caller never matches axes by hand.
    pub fn at(&self, region: &Region) -> Tile<T> {
        let tile_kind = match &self.tile_kind {
            TileKind::Gmem(g) => TileKind::new_Gmem(g.at(region, comptime!(self.space.clone()))),
            TileKind::Smem(g) => TileKind::new_Smem(g.at(region, comptime!(self.space.clone()))),
            TileKind::TmaGmem(t) => {
                TileKind::new_TmaGmem(t.at(region, comptime!(self.space.clone())))
            }
            TileKind::Procedural(p) => {
                TileKind::new_Procedural(p.at(region, comptime!(self.space.clone())))
            }
            // A plane tile has nothing to window: pass it through. Legal only where the level
            // cuts nothing on m/n (a k-step walk); a cutting level would alias every region
            // onto the one tile.
            TileKind::PlaneTile(t) => {
                comptime!(assert!(
                    !self.space.cuts_tiles(),
                    "Tile::at: a level that cuts tiles cannot select into a single plane \
                     tile (it needs a partition, or a memory output)"
                ));
                TileKind::new_PlaneTile(t.clone())
            }
            // A partition selects under comptime coordinates (an unrolled walk folds regions
            // to constants): each region owns a `sub_m × sub_n` block. An uncut level selects
            // the whole partition; a 1×1 block is the tile itself. A runtime region passes the
            // partition through whole, legal only on an uncut k-step level (the walk below
            // then selects statically).
            TileKind::PlanePartition(p) => {
                let rank = comptime!(self.space.rank());
                let a0 = comptime!(self.space.axis_at(rank - 2));
                let a1 = comptime!(self.space.axis_at(rank - 1));
                // A single-tile static axis (k-step, no m/n cut) folds to constant `0`, so a
                // cut axis takes its constant digit and an uncut one selects the whole
                // partition. A `Dynamic` axis (top level only) stays runtime, yielding `None`.
                let mi = if comptime!(self.space.single_static_tile(a0)) {
                    comptime!(Some(0u64))
                } else {
                    region.coord(a0).constant()
                };
                let ni = if comptime!(self.space.single_static_tile(a1)) {
                    comptime!(Some(0u64))
                } else {
                    region.coord(a1).constant()
                };
                match comptime!(mi.zip(ni)) {
                    Some((c0, c1)) => {
                        let (sub_m, sub_n) = comptime!({
                            let (cm, cn) = (self.space.count(a0), self.space.count(a1));
                            assert!(
                                p.m_tiles.is_multiple_of(cm) && p.n_tiles.is_multiple_of(cn),
                                "Tile::at: the level's grid must divide the partition"
                            );
                            (p.m_tiles / cm, p.n_tiles / cn)
                        });
                        let mi = comptime!(c0 as usize * sub_m);
                        let ni = comptime!(c1 as usize * sub_n);
                        if comptime!(sub_m == 1 && sub_n == 1) {
                            TileKind::new_PlaneTile(p.at(mi, ni))
                        } else {
                            TileKind::new_PlanePartition(p.window(mi, ni, sub_m, sub_n))
                        }
                    }
                    // A runtime coordinate reaches here only from a `Dynamic` (top, instance)
                    // level, which cuts nothing on m/n and passes the whole partition down to
                    // the static levels below. A rolled *cut* would be a caller bug.
                    None => {
                        comptime!(assert!(
                            !self.space.cuts_tiles(),
                            "Tile::at: a level that cuts a partition must be \
                             walked with compile-time coordinates (an unrolled walk)"
                        ));
                        TileKind::new_PlanePartition(p.clone())
                    }
                }
            }
        };
        Tile::<T> {
            tile_kind,
            space: comptime!(self.space.divide()),
        }
    }

    /// Whether this tile has a buffer bound to read an extent off: memory and a tensor map carry
    /// the tensor's shape, a resident fragment carries nothing but its cells. The kinds
    /// [`runtime_extent`](Tile::runtime_extent) can answer for, so the two match the same way.
    fn bounded(&self) -> comptime_type!(bool) {
        match &self.tile_kind {
            TileKind::Gmem(_) | TileKind::Smem(_) | TileKind::TmaGmem(_) => comptime!(true),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) | TileKind::Procedural(_) => {
                comptime!(false)
            }
        }
    }

    /// This operand's runtime logical size along `axis`, read off the [`bound`](MemData) folded
    /// from the tensor shape, and the source of a [`Dynamic`](crate::Extent) axis's tile count.
    /// Only an axis this tile [`witnesses`](Tile::witnesses) has one.
    pub(crate) fn runtime_extent(&self, #[comptime] axis: Axis) -> usize {
        let projection = self.projection();
        let p = comptime!(bound_position(&projection, axis));
        let raw = match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => g.window.bound.at(p).fcast::<usize>(),
            TileKind::TmaGmem(t) => t.bound[p].fcast::<usize>(),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::runtime_extent: a plane tile has no extent")
            }
            TileKind::Procedural(_) => {
                panic!("Tile::runtime_extent: a procedural tile has no extent")
            }
        };
        // `bound` is a line count on the vectorized innermost axis; the walk divides by
        // conceptual edges, so return line count × width.
        let last = comptime!(projection.physical_rank() - 1);
        let w = self.vector_size();
        comptime!(if p == last { w } else { 1usize }) * raw
    }

    /// The runtime space to walk this tile *alone*: [`witnessed_space`] with no other operand to
    /// ask, so every `Dynamic` axis must be one this tile itself
    /// [`witnesses`](Tile::witnesses). An operation over several operands sizes its space from all
    /// of them instead, which is what lets a gathered operand ride an axis it cannot answer for.
    pub fn runtime_space(&self) -> Space {
        witnessed_space(comptime!(self.space.clone()), self, self, self)
    }

    /// Zero this tile: `mma` accumulates over whatever is there, so a routine whose contract is
    /// `out = A·B` zeroes first. Same shape as [`mma`](Tile::mma): a final tile clears its store,
    /// a level walks and recurses (each region clears exactly the windows it owns; a fragment
    /// output takes the unrolled walk, memory the compact loop).
    pub fn zero(&mut self) {
        match comptime!(self.space.partitioner().clone()) {
            Partitioner::Final => match &mut self.tile_kind {
                TileKind::Gmem(d) | TileKind::Smem(d) => d.zero(),
                TileKind::PlaneTile(t) => t.zero(),
                TileKind::PlanePartition(p) => p.zero(),
                TileKind::TmaGmem(_) => panic!("Tile::zero: a tma source is not writable"),
                TileKind::Procedural(_) => panic!("Tile::zero: a procedural tile is not writable"),
            },
            Partitioner::Level(_) => {
                let unroll = self.tile_kind.static_level(comptime!(self.space.clone()));
                for region in Walk::over(self.runtime_space()).with_unroll(unroll) {
                    let mut sub = self.at(&region);
                    sub.zero();
                }
            }
        }
    }

    /// Seed this tile with `monoid`'s identity, so a fold under it starts from a value folding it
    /// in leaves unchanged. `Sum` goes through [`zero`](Tile::zero), which every accumulator form
    /// can do; the other monoids need a real value and so reach only [`init`](Tile::init)'s forms.
    pub(crate) fn init_identity(&mut self, #[comptime] monoid: Monoid) {
        match comptime!(monoid) {
            Monoid::Sum => self.zero(),
            Monoid::Prod | Monoid::Max | Monoid::Min => self.init(Monoid::identity::<T>(monoid)),
        }
    }

    /// Initialize this tile with `val`. Same shape as [`zero`](Tile::zero).
    pub fn init(&mut self, val: T) {
        match comptime!(self.space.partitioner().clone()) {
            Partitioner::Final => match &mut self.tile_kind {
                TileKind::Gmem(d) | TileKind::Smem(d) => d.init(val),
                TileKind::PlaneTile(t) => t.init(val),
                TileKind::PlanePartition(p) => p.init(val),
                TileKind::TmaGmem(_) => panic!("Tile::init: a tma source is not writable"),
                TileKind::Procedural(_) => panic!("Tile::init: a procedural tile is not writable"),
            },
            Partitioner::Level(_) => {
                let unroll = self.tile_kind.static_level(comptime!(self.space.clone()));
                for region in Walk::over(self.runtime_space()).with_unroll(unroll) {
                    let mut sub = self.at(&region);
                    sub.init(val);
                }
            }
        }
    }

    /// The window as one dense run of `Vector<T, W>` lines (`W` the store's
    /// own width): index `i` reads line `origin + i`, one add and no layout
    /// walk. See [`MemData::dense_lines`] for the (caller-owned) contiguity
    /// contract; the streaming fold's operands satisfy it by construction.
    pub fn dense<W: Size>(&self) -> &[Vector<T, W>] {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.dense_lines::<W>(),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::dense: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::dense: a tma source has no element view"),
            TileKind::Procedural(_) => panic!("Tile::dense: a procedural tile has no memory view"),
        }
    }

    /// The mutable twin of [`dense`](Tile::dense).
    pub(crate) fn dense_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        match &mut self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.dense_lines_mut::<W>(),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::dense_mut: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::dense_mut: a tma source is not writable"),
            TileKind::Procedural(_) => panic!("Tile::dense_mut: a procedural tile is not writable"),
        }
    }

    /// Move `src` into `self`, the physical pairing picking the instruction that does it. This is
    /// the move itself, not [`StageSource::Transport`](crate::StageSource), which is a staging
    /// slot's plan to make one. A partition source is matched first because it needs the whole
    /// destination tile, which the pairing match below would keep borrowed.
    pub fn copy_from(&mut self, src: &Tile<T>) {
        // Bound before the match, which borrows the kind: a memory fill needs the logical space
        // both sides carry (a gathered source is addressed per axis).
        let space = comptime!(self.space.clone());
        match &src.tile_kind {
            TileKind::PlanePartition(s) => s.drain_into(self),
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => match (&mut self.tile_kind, &src.tile_kind) {
                (TileKind::PlanePartition(d), TileKind::Gmem(_) | TileKind::Smem(_)) => {
                    d.fill_from(src)
                }
                (TileKind::PlaneTile(d), TileKind::Gmem(_) | TileKind::Smem(_)) => {
                    d.load_window(src)
                }
                (TileKind::Gmem(d) | TileKind::Smem(d), TileKind::PlaneTile(s)) => {
                    s.store_window(d, space)
                }
                (TileKind::Smem(d), TileKind::TmaGmem(s)) => s.load_into(d),
                (TileKind::Gmem(d) | TileKind::Smem(d), TileKind::Gmem(s) | TileKind::Smem(s)) => {
                    d.fill_from(s, space)
                }
                (TileKind::Gmem(d) | TileKind::Smem(d), TileKind::Procedural(s)) => {
                    d.fill_procedural(s, space)
                }
                (TileKind::PlaneTile(_), TileKind::PlaneTile(_)) => {
                    panic!("Tile::copy_from: plane tile to plane tile cast not wired")
                }
                _ => panic!("Tile::copy_from: unsupported kind pairing"),
            },
        }
    }

    /// Drain a resident accumulator into memory `dst`, casting `T` down to `dst`'s element type.
    /// [`copy_from`](Self::copy_from) cannot: its transports move bytes and stay same-type, but a
    /// register accumulator is wider than the output it writes. Crate-internal, because closing an
    /// accumulator's scope is the scope's own business.
    pub(crate) fn drain_cast_into<Out: Numeric>(&self, dst: &mut Tile<Out>) {
        match &self.tile_kind {
            TileKind::PlanePartition(s) => s.drain_cast_into(dst),
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                panic!("Tile::drain_cast_into: only a partition drains with a cast")
            }
        }
    }

    /// Whether one factor tap lands inside the input axes that factor moves. The physical position
    /// is already stepped from the row's hoisted anchor; the memory window then checks only the
    /// physical carriers of `axis`, avoiding rebuilding the projection per tap.
    pub(crate) fn separable_physical_tap_in_bounds(
        &self,
        pos: &CoordsDyn,
        #[comptime] axis: Axis,
    ) -> bool {
        match &self.tile_kind {
            TileKind::Gmem(data) => {
                let projection = comptime!(data.projection.clone());
                if comptime!(projection.addresses(axis)) {
                    let carriers = comptime!(projection.carriers(axis));
                    let mut valid = true;
                    #[unroll]
                    for c in 0..comptime!(carriers.len()) {
                        let pa = comptime!(carriers[c]);
                        if comptime!(
                            data.window.boundaries.get(pa).copied().flatten()
                                == Some(Boundary::Zero)
                        ) {
                            valid = valid && data.window.axis_in_bounds(pos[pa], pa);
                        }
                    }
                    valid
                } else {
                    true.runtime()
                }
            }
            // A staged operand answers against the window it was *filled from*: the fill wrote
            // the boundary's value wherever a tap fell outside, and this window no longer says
            // which cells those were. A stage recorded with no source window was never gathered,
            // so nothing it holds came from a boundary and every tap is in bounds.
            TileKind::Smem(data) => {
                let projection = comptime!(data.projection.clone());
                #[comptime]
                match &data.source_window {
                    ComptimeOption::Some(source) => {
                        if comptime!(projection.logical_axes().contains(&axis)) {
                            let carriers = comptime!(projection.carriers(axis));
                            let mut valid = true;
                            #[unroll]
                            for c in 0..comptime!(carriers.len()) {
                                let pa = comptime!(carriers[c]);
                                if comptime!(
                                    source.boundaries.get(pa).copied().flatten()
                                        == Some(Boundary::Zero)
                                ) {
                                    valid = valid
                                        && source.axis_in_bounds(
                                            data.window.origin.at(pa),
                                            pos[pa],
                                            pa,
                                        );
                                }
                            }
                            valid
                        } else {
                            true.runtime()
                        }
                    }
                    ComptimeOption::None => true.runtime(),
                }
            }
            TileKind::Procedural(data) => data.axis_in_bounds(pos, axis),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) | TileKind::TmaGmem(_) => {
                panic!(
                    "Tile::separable_physical_tap_in_bounds: a separable gather needs an addressable rhs"
                )
            }
        }
    }
}

impl<T: Numeric> Tile<T> {
    pub(crate) fn factor_dependencies(
        &self,
        _factors: Option<usize>,
        _row: Axis,
        _col: Axis,
    ) -> comptime_type!(Option<Vec<(bool, bool)>>) {
        unexpanded!()
    }
}

impl<T: Numeric> TileExpand<T> {
    pub(crate) fn __expand_factor_dependencies_method(
        &self,
        scope: &Scope,
        factors: Option<usize>,
        row: Axis,
        col: Axis,
    ) -> Option<Vec<(bool, bool)>> {
        factors.map(|factors| match &self.tile_kind {
            TileKindExpand::Procedural(data) => (0..factors)
                .map(|f| {
                    (
                        data.__expand_factor_reads_axis_method(scope, f, row),
                        data.__expand_factor_reads_axis_method(scope, f, col),
                    )
                })
                .collect(),
            TileKindExpand::Gmem(_)
            | TileKindExpand::Smem(_)
            | TileKindExpand::PlaneTile(_)
            | TileKindExpand::PlanePartition(_)
            | TileKindExpand::TmaGmem(_) => (0..factors).map(|_| (true, true)).collect(),
        })
    }
}

/// `space` with each [`Dynamic`](crate::Extent) axis sized by the first of `a`, `b`, `c` that
/// [`witnesses`](Tile::witnesses) it, which is how an operation turns its comptime space into the
/// runtime one [`Walk::over`](crate::Walk) takes. A fully-`Static` space short-circuits to no
/// runtime sizes. One tile may stand for all three ([`runtime_space`](Tile::runtime_space)).
#[cube]
pub fn witnessed_space<A: Numeric, B: Numeric, C: Numeric>(
    #[comptime] space: Space,
    a: &Tile<A>,
    b: &Tile<B>,
    c: &Tile<C>,
) -> Space {
    let mut sizes = Sequence::<usize>::new();
    if comptime!(!space.is_static()) {
        #[unroll]
        for p in 0..comptime!(space.rank()) {
            let axis = comptime!(space.axis_at(p));
            // `sizes` is positional, so every axis pushes, but only a `Dynamic` one is ever read
            // back ([`Extents::count`] folds a `Static` axis to its comptime extent). Fold it here
            // too rather than asking an operand: one `Dynamic` axis must not make the `Static`
            // ones unreadable on a tile that has no bound to answer with.
            let size = match comptime!(space.extent_raw(axis)) {
                Extent::Static(n) => comptime!(n).runtime(),
                Extent::Dynamic => {
                    let by_a = a.witnesses(axis);
                    let by_b = b.witnesses(axis);
                    let by_c = c.witnesses(axis);
                    if comptime!(by_a) {
                        a.runtime_extent(axis)
                    } else if comptime!(by_b) {
                        b.runtime_extent(axis)
                    } else if comptime!(by_c) {
                        c.runtime_extent(axis)
                    } else {
                        panic!(
                            "witnessed_space: {axis:?} is Dynamic and no operand states its size; \
                             every operand spanning it gathers over it, holds it Static, or is a \
                             fragment. Keep it Static in the kernel space, or give the operation \
                             an operand that maps it identically"
                        )
                    }
                }
            };
            sizes.push(size);
        }
    }
    Space::with_sizes(space, sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: Axis = Axis(0);
    const B: Axis = Axis(1);

    /// The discrimination the operation space rests on: a bound is an axis's own extent only when
    /// one dim carries that axis alone. The two ways it stops being one are a gather, whose dim
    /// holds the receptive field its axes reach over, and storage tiling, which splits the extent
    /// across dims so no single bound is it.
    #[test]
    fn bound_states_wants_one_dim_carrying_the_axis_alone() {
        let direct = Projection::direct(&[A, B]);
        assert_eq!(bound_states(&direct, A), Some(0));
        assert_eq!(bound_states(&direct, B), Some(1));

        let gathered = Projection::new(&[A, B], &[PhysicalAxisMap::affine(&[(A, 1), (B, 1)])]);
        assert_eq!(bound_states(&gathered, A), None);
        assert_eq!(bound_states(&gathered, B), None);

        let tiled = Projection::tiled(&[A, B], StorageTiling::per_axis(&[1, 2]));
        assert_eq!(bound_states(&tiled, A), Some(0));
        assert_eq!(bound_states(&tiled, B), None);
    }
}
