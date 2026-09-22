//! The [`Tile`]: one operand's data as a [`TileKind`] backing store, plus the comptime
//! [`Space`] it projects. Structure only; each store's own data and leaves live in its file
//! ([`mem`](crate::Memory), [`cmma`](crate::CmmaData), [`tma`](crate::TmaData)).

use cubecl::{prelude::*, std::tensor::layout::CoordsDyn, unexpanded};

use cubecl::zspace::SmallVec;

use crate::*;

/// A tile's backing store. Every variant is lifetime-free (a `Box<[T]>` or a
/// [`cmma::Matrix`](cubecl::cmma::Matrix)); [`view`](Tile::view) rebuilds a borrowed view on
/// demand. `Clone` copies the handle, not the cells: sound only where nothing rewrites the buffer.
// A kernel value: every variant is what the kernel holds, and a boxed variant is not a cube
// type, so the size difference stays.
#[allow(clippy::large_enum_variant)]
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum TileKind<T: Numeric> {
    /// An addressable buffer with a window, in global or shared memory
    /// ([`address`](Memory::address)).
    Memory(Memory<T>),
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
    /// A tile the plane holds in its lanes, one line to a lane, shared by shuffle and read at
    /// coordinates ([`Lanes`]).
    Lanes(Lanes<T>),
}

/// One operand's data: a runtime [`TileKind`] backing store and the comptime [`Space`] it
/// projects. `T` is the element the tile serves and computes in; its physical vector width is a
/// [`TileKind`] storage detail, read back with [`vector_size`](Tile::vector_size).
///
/// An operand's kind is what the kernel made it (a memory window, a stage, a plane fragment), so
/// operands that disagree meet the leaf's kind-pairing panics.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Tile<T: Numeric> {
    pub kind: TileKind<T>,
    /// Where this tile sits in its partitioning: its box, its depth, its levels.
    #[cube(comptime)]
    pub place: Placement,
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// `kind` at `place`: the one constructor, so a kind and its place are never apart.
    pub fn new(kind: TileKind<T>, #[comptime] place: Placement) -> Tile<T> {
        Tile::<T> { kind, place }
    }

    /// This tile's own box: the axes it spans and their extents at this depth.
    pub fn space(&self) -> comptime_type!(Space) {
        comptime!(self.place.space.clone())
    }

    /// This tile's memory store, for a reader that addresses bytes. `site` names the reader in
    /// the refusal: a plane tile, a tensor-map source, a recipe and the plane's lanes have none.
    pub(crate) fn mem(&self, #[comptime] site: &str) -> &Memory<T> {
        match &self.kind {
            TileKind::Memory(g) => g,
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::{site}: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::{site}: a tma source has no element view"),
            TileKind::Procedural(_) | TileKind::Lanes(_) => {
                panic!("Tile::{site}: a procedural tile and the plane's lanes have no memory view")
            }
        }
    }

    /// The mutable twin of [`mem`](Tile::mem).
    pub(crate) fn mem_mut(&mut self, #[comptime] site: &str) -> &mut Memory<T> {
        match &mut self.kind {
            TileKind::Memory(g) => g,
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::{site}: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::{site}: a tma source is not writable"),
            TileKind::Procedural(_) | TileKind::Lanes(_) => {
                panic!("Tile::{site}: a procedural tile and the plane's lanes are not writable")
            }
        }
    }

    /// Evaluate a procedural tile at scalar logical coordinates relative to its current region.
    pub fn procedural_value(&self, pos: Coords<u32>) -> T {
        match &self.kind {
            TileKind::Procedural(data) => data.evaluate(&pos, comptime!(self.place.space.clone())),
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lanes(_) => panic!("Tile::procedural_value: tile is not procedural"),
        }
    }

    /// Who moves this operand's bytes into a stage: the cube's units, the TMA engine, or nobody
    /// (a cooperative evaluation). How it is stored does not enter into it, so a storage-tiled
    /// buffer and a plain one both copy. A plane fragment has no bytes to move and panics here.
    pub fn delivery(&self) -> comptime_type!(Delivery) {
        match &self.kind {
            TileKind::Memory(_) => comptime!(Delivery::Copy),
            TileKind::TmaGmem(_) => comptime!(Delivery::Tma),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::delivery: a resident fragment is not a stage source")
            }
            // A procedural source is cooperatively materialized into its stage.
            TileKind::Procedural(_) | TileKind::Lanes(_) => comptime!(Delivery::Procedural),
        }
    }

    /// The runtime map (dynamic coefficients and origin phase residues) of this tile.
    pub(crate) fn runtime_map(&self) -> RuntimeMap {
        match &self.kind {
            TileKind::Memory(g) => g.map.clone(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lanes(_) => RuntimeMap::integral(comptime!(self.place.space.rank())),
        }
    }

    /// The launch's cube size this operand was bound with, `0` when unknown; what a stage filled
    /// from it emits its fill straight-line by.
    pub(crate) fn units(&self) -> comptime_type!(usize) {
        match &self.kind {
            TileKind::Memory(d) => comptime!(d.access.units),
            TileKind::TmaGmem(t) => comptime!(t.units),
            TileKind::Procedural(_)
            | TileKind::Lanes(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_) => {
                comptime!(0)
            }
        }
    }

    /// Physical vectorization of the backing store: the `Vector<T, vector_size>` line
    /// width the leaf reconstructs. A launched memory tile carries its operand's vector
    /// size; a cmma fragment and a tma source are scalar (`1`).
    pub fn vector_size(&self) -> comptime_type!(usize) {
        match &self.kind {
            TileKind::Memory(d) => d.store.vector_size,
            TileKind::Lanes(c) => c.line(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(1usize)
            }
        }
    }

    /// Ask this accumulator to start from `init_from`, and answer what actually took, so a kind
    /// that cannot take the request is listed nowhere else.
    ///
    /// A folding destination always answers [`Identity`](InitFrom::Identity): its cells belong to
    /// several instances, so the buffer must already hold the fold's identity (the launch's
    /// obligation), and `Cell` would seed a cell the siblings are folding into.
    pub(crate) fn request_init_from(
        &mut self,
        #[comptime] init_from: InitFrom,
    ) -> comptime_type!(InitFrom) {
        match &mut self.kind {
            TileKind::Memory(d) => {
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
            | TileKind::Procedural(_)
            | TileKind::Lanes(_) => {
                comptime!(InitFrom::Cell)
            }
        }
    }

    /// This operand's decode site ([`DequantAt`]). A tile with nothing to decode answers
    /// [`DequantAt::Load`]: served and stored are one element, so its load already delivers what
    /// the read wants. A tma source is never quantized and answers the same; a bulk copy cannot.
    pub(crate) fn dequant_at(&self) -> comptime_type!(DequantAt) {
        match &self.kind {
            TileKind::Memory(d) => d.dequant_at(),
            TileKind::TmaGmem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::Procedural(_)
            | TileKind::Lanes(_) => {
                comptime!(DequantAt::Load)
            }
        }
    }

    /// How this tile's values sit in memory; see [`Memory::packing`]. A resident fragment, a tma
    /// source and a procedural tile are never quantized.
    pub(crate) fn packing(&self) -> comptime_type!(Packing) {
        match &self.kind {
            TileKind::Memory(d) => d.packing(),
            TileKind::Lanes(c) => c.packing(),
            TileKind::TmaGmem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::Procedural(_) => {
                comptime!(Packing::Plain)
            }
        }
    }

    /// Whether this tile's buffer is addressed through a mapping whose windows may *overlap*, so
    /// only the N-D surface ([`nd`](Tile::nd)) describes it and no window is dense. False for
    /// [`direct`](Projection::direct) and for a [partition](Composition::Disjoint), a bijection.
    pub fn gathered(&self) -> comptime_type!(bool) {
        let projection = self.projection();
        comptime!(projection.composition() == Composition::Overlapping)
    }

    /// Whether this tile evaluates values from coordinates instead of a backing buffer.
    pub(crate) fn is_procedural(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Procedural(_) => comptime!(true),
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lanes(_) => comptime!(false),
        }
    }

    /// Whether this tile is backed by shared memory, which is what a `sync_cube()` between two
    /// units' accesses actually orders: the barrier covers the workgroup address space, so a
    /// tile a cube communicates *through* has to live there and not in a global buffer.
    pub(crate) fn is_shared(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(m) => comptime!(m.address == AddressSpace::Shared),
            TileKind::Procedural(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lanes(_) => comptime!(false),
        }
    }

    /// The factorization this tile's values state, if any: `Some(n)` for a recipe presenting `n`
    /// separable factors, `None` for a buffer or a recipe that only answers as a whole. Rank one is
    /// `Some(1)`, deliberately distinct from `None`: a consumer can still exploit it.
    pub(crate) fn factors(&self) -> comptime_type!(Option<usize>) {
        match &self.kind {
            TileKind::Procedural(data) => data.factors(),
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lanes(_) => comptime!(None),
        }
    }

    /// This operand's window sign, for a stage recording where it was filled from.
    pub(crate) fn window_signed(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(d) => comptime!(d.window.signed),
            _ => comptime!(false),
        }
    }

    /// This operand's per-axis boundary handling, read by a stage filled from it: the stage's own
    /// list is empty (it never overhangs its buffer), so the padding policy has to come from here.
    pub(crate) fn window_boundaries(
        &self,
    ) -> comptime_type!(SmallVec<[Option<Boundary>; Space::MAX_RANK]>) {
        match &self.kind {
            TileKind::Memory(d) => comptime!(d.window.boundaries.clone()),
            _ => comptime!(SmallVec::new()),
        }
    }

    /// The physical boundaries factor-local normalization tests. A shared-memory tile preserves
    /// these on its source window even though its resident window itself no longer overhangs.
    pub(crate) fn separable_boundaries(
        &self,
    ) -> comptime_type!(SmallVec<[Option<Boundary>; Space::MAX_RANK]>) {
        match &self.kind {
            // A shared stage preserves them on its source window: its own window no longer
            // overhangs, and a stage with no source window was never gathered.
            TileKind::Memory(d) =>
            {
                #[comptime]
                match &d.source_window {
                    ComptimeOption::Some(source) => comptime!(source.boundaries.clone()),
                    ComptimeOption::None => comptime!(match d.address {
                        AddressSpace::Global => d.window.boundaries.clone(),
                        AddressSpace::Shared => SmallVec::new(),
                    }),
                }
            }
            TileKind::Procedural(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lanes(_) => comptime!(SmallVec::new()),
        }
    }

    /// The separable factor-normalization request, if one was attached to this procedural tile.
    /// Backed tiles answer `None`, as they have no factor evaluation for the gather leaf to alter.
    pub(crate) fn factor_normalization(
        &self,
    ) -> comptime_type!(Option<(TapMask, DivGuard, Space)>) {
        match &self.kind {
            TileKind::Procedural(data) => comptime!(data.normalization.clone()),
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lanes(_) => comptime!(None),
        }
    }

    /// One factor of a separable recipe, evaluated at `pos`. Only the coordinate along the axis
    /// that factor reads matters, which is what lets the contraction walk it in 1-D. Asking
    /// [`factors`](Tile::factors) first is the whole precondition.
    pub(crate) fn separable_factor(&self, pos: CoordsDyn, #[comptime] factor: usize) -> T {
        match &self.kind {
            TileKind::Procedural(data) => {
                data.evaluate_factor_dyn(&pos, factor, comptime!(self.place.space.clone()))
            }
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lanes(_) => {
                panic!(
                    "Tile::separable_factor: a tile read from a buffer states no factorization, \
                     so `factors` answered `None` and there is no factor {factor} to evaluate"
                )
            }
        }
    }

    /// Whether this tile can state `axis`'s runtime size: it spans it [`Dynamic`](crate::Extent),
    /// has a buffer to read a bound off, and the bound is the axis's own extent ([`bound_states`]).
    /// An operation sizes a `Dynamic` axis from any operand witnessing it ([`witnessed_space`]).
    pub fn witnesses(&self, #[comptime] axis: Axis) -> comptime_type!(bool) {
        let bounded = self.bounded();
        let projection = self.projection();
        comptime!(
            bounded
                && self.place.space.contains(axis)
                && self.place.space.is_dynamic(axis)
                && Witness::new(&projection, axis).is_some()
        )
    }

    /// How this tile's logical axes address its buffer's physical ones. A fragment and a tma source
    /// have no buffer to project onto, so they answer [`direct`](Projection::direct) over their own
    /// space, which is what every non-gather operand carries anyway.
    pub(crate) fn projection(&self) -> comptime_type!(Projection) {
        match &self.kind {
            TileKind::Memory(g) => comptime!(g.projection.clone()),
            // The lanes keep the projection of the operand they stage: which axes one line
            // holds whole is that operand's fact.
            TileKind::Lanes(c) => c.projection(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(Projection::direct_over(&self.place.space))
            }
        }
    }

    /// This operand's window placed at `from` on `axis`, reading no further than `until`, both
    /// counted in that axis's own elements. A region names a whole tile of an axis; this names an
    /// element and where reads stop, both runtime, as a packed sequence's start and length are.
    ///
    /// `until` arms [`Boundary::Zero`] on the axis, so a last tile overrunning the range reads zero
    /// there rather than the next sequence's values.
    pub fn within(&self, #[comptime] axis: Axis, from: usize, until: usize) -> Tile<T> {
        let tile_kind = match &self.kind {
            TileKind::Memory(g) => TileKind::new_Memory(g.within(axis, from, until)),
            TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lanes(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_) => panic!(
                "Tile::within: only a memory operand carries a window to place; a fragment, a \
                 tensor-map source and a recipe have none"
            ),
        };
        Tile::new(tile_kind, comptime!(self.place.clone()))
    }

    /// Window this tile down to `region`, no copy: the steps of the region's path below this
    /// tile's depth, applied in turn. Each tile projects a step onto its own axes, so `lhs ∈ {M,K}`
    /// and `out ∈ {M,N}` line up without the caller matching axes; root and window read one region.
    pub fn at(&self, region: &Region) -> Tile<T> {
        let skip = comptime!({
            let path = &region.path;
            assert!(
                path.base() <= self.place.depth && self.place.depth < path.depth(),
                "Tile::at: this tile sits {} levels down its nest, and the region's path runs \
                 from depth {} through {} levels; state the loops the tile is missing above it",
                self.place.depth,
                path.base(),
                path.len()
            );
            self.place.depth - path.base()
        });
        self.at_from(region, skip)
    }

    /// The path's steps from the `i`-th down, applied in turn.
    fn at_from(&self, region: &Region, #[comptime] i: usize) -> Tile<T> {
        let sub = self.at_step(&region.step(i));
        if comptime!(i + 1 == region.path.len()) {
            sub
        } else {
            sub.at_from(region, comptime!(i + 1))
        }
    }

    /// One level down: window this tile to `step`'s box.
    fn at_step(&self, step: &Step) -> Tile<T> {
        let tile_kind = match &self.kind {
            TileKind::Memory(g) => {
                TileKind::new_Memory(g.at(step, comptime!(self.place.space.clone())))
            }
            TileKind::TmaGmem(t) => {
                TileKind::new_TmaGmem(t.at(step, comptime!(self.place.space.clone())))
            }
            TileKind::Procedural(p) => {
                TileKind::new_Procedural(p.at(step, comptime!(self.place.space.clone())))
            }
            TileKind::Lanes(c) => {
                TileKind::new_Lanes(c.at(step, comptime!(self.place.space.clone())))
            }
            // A plane tile has nothing to window: pass it through. Legal only where the level
            // cuts nothing on m/n (a k-step walk); a cutting level would alias every region
            // onto the one tile.
            TileKind::PlaneTile(t) => {
                comptime!(assert!(
                    !MatrixGrid::new(&step.level, &self.place.space).cuts(),
                    "Tile::at: a level that cuts tiles cannot select into a single plane \
                     tile (it needs a partition, or a memory output)"
                ));
                TileKind::new_PlaneTile(t.clone())
            }
            TileKind::PlanePartition(p) => p.at_step(step, comptime!(self.place.space.clone())),
        };
        Tile::<T> {
            kind: tile_kind,
            place: comptime!(self.place.at_step(&step.level)),
        }
    }

    /// Whether this tile has a buffer bound to read an extent off: memory and a tensor map carry
    /// the tensor's shape, a resident fragment carries nothing but its cells. The kinds
    /// [`runtime_extent`](Tile::runtime_extent) can answer for, so the two match the same way.
    fn bounded(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(_) | TileKind::TmaGmem(_) => comptime!(true),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::Procedural(_)
            | TileKind::Lanes(_) => {
                comptime!(false)
            }
        }
    }

    /// This operand's runtime logical size along `axis`, read off the [`bound`](Memory) folded
    /// from the tensor shape, and the source of a [`Dynamic`](crate::Extent) axis's tile count.
    /// Only an axis this tile [`witnesses`](Tile::witnesses) has one.
    pub(crate) fn runtime_extent(&self, #[comptime] axis: Axis) -> usize {
        let projection = self.projection();
        let p = comptime!(
            Witness::new(&projection, axis)
                .unwrap_or_else(|| panic!(
                    "Tile::runtime_extent: no bound of this operand is {axis:?}'s own extent (it \
                     gathers over it, or splits it across storage fragments); ask an operand that \
                     witnesses it"
                ))
                .dim()
        );
        let raw = match &self.kind {
            TileKind::Memory(g) => g.window.bound.at(p).retyped::<usize>(),
            TileKind::TmaGmem(t) => t.bound[p].retyped::<usize>(),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::runtime_extent: a plane tile has no extent")
            }
            TileKind::Procedural(_) | TileKind::Lanes(_) => {
                panic!(
                    "Tile::runtime_extent: a procedural tile and the plane's lanes have no extent"
                )
            }
        };
        // `bound` is a line count on the vectorized innermost axis; the walk divides by
        // conceptual edges, so return line count × width.
        let last = comptime!(projection.physical_rank() - 1);
        let w = self.vector_size();
        comptime!(if p == last { w } else { 1usize }) * raw
    }

    /// This tile's own box, as the runtime space a loop walks: its axes alone, a dynamic one
    /// sized off its buffer.
    pub(crate) fn runtime_space(&self) -> Space {
        witnessed_space(comptime!(self.place.space.clone()), self, self, self)
    }

    /// The regions of the level below this tile over its own box, its axes alone: a loop over one
    /// operand's windows steps nothing the operand does not span, where the kernel's space would.
    /// What `for plane in tile` iterates, as a value.
    pub fn walk(&self) -> Walk {
        let space = self.runtime_space();
        Region::rooted(
            &space,
            comptime!(self.place.levels.clone()),
            comptime!(self.place.depth),
        )
        .walk()
    }

    /// The regions of `level` over this tile's own box, a level of the kernel's own rather than
    /// the partitioning's next ([`walk`](Tile::walk)). The regions sit one level below this
    /// tile's depth.
    pub fn over(&self, #[comptime] level: &Level) -> Walk {
        let space = self.runtime_space();
        Walk::of(
            &space,
            comptime!(level.clone()),
            Region::rooted(
                &space,
                comptime!(self.place.levels.clone()),
                comptime!(self.place.depth),
            ),
        )
    }

    /// Zero this tile: `mma` accumulates over whatever is there, so a routine whose contract is
    /// `out = A·B` zeroes first. Covers the whole window this tile holds (every cube unit for
    /// memory, every fragment for a partition); the kernel's own loops split it across levels.
    pub fn zero(&mut self) {
        match &mut self.kind {
            TileKind::Memory(d) => d.zero(),
            TileKind::PlaneTile(t) => t.zero(),
            TileKind::PlanePartition(p) => p.zero(),
            TileKind::TmaGmem(_) => panic!("Tile::zero: a tma source is not writable"),
            TileKind::Procedural(_) | TileKind::Lanes(_) => {
                panic!("Tile::zero: a procedural tile and the plane's lanes are not writable")
            }
        }
    }

    /// Seed this tile with `monoid`'s identity, so a fold under it starts from a value folding it
    /// in leaves unchanged. `Sum` goes through [`zero`](Tile::zero), which every accumulator form
    /// can do; the other monoids need a real value and so reach only [`init`](Tile::init)'s forms.
    pub fn init_identity(&mut self, #[comptime] monoid: Monoid) {
        match comptime!(monoid) {
            Monoid::Sum => self.zero(),
            Monoid::Prod | Monoid::Max | Monoid::Min => self.init(Monoid::identity::<T>(monoid)),
        }
    }

    /// The one value this tile holds, read through whatever packing its binding states. What a
    /// scale covering everything is: a tile every axis of which it spans at an extent of one.
    pub(crate) fn only(&self) -> T {
        let axes = comptime!(MatrixAxes::trailing(&self.place.space));
        let matrix = self.matrix_packed::<Const<1>>(axes, 0usize);
        let origin = 0u32.runtime();
        matrix.read((origin, origin)).extract(0usize)
    }

    /// Multiply every partial this accumulator holds by the one value `factor` carries, or by
    /// nothing where no factor was bound, which is multiplying by one.
    ///
    /// A scale covering everything the accumulator sums belongs here rather than on the terms: one
    /// multiply per cell instead of one per value read. A scale that does *not* cover everything
    /// summed cannot come here at all and rides its factor instead ([`Tile::scaled`]).
    pub fn scale<S: Numeric>(&mut self, factor: &ComptimeOption<Tile<S>>) {
        #[comptime]
        match factor {
            ComptimeOption::Some(factor) => {
                let value = T::cast_from(factor.only());
                match &mut self.kind {
                    TileKind::PlaneTile(t) => t.scale(value),
                    TileKind::PlanePartition(p) => p.scale(value),
                    TileKind::Memory(_) => panic!(
                        "Tile::scale: a memory tile is scaled by the cube, not by one unit (Tile::mul)"
                    ),
                    TileKind::TmaGmem(_) | TileKind::Procedural(_) | TileKind::Lanes(_) => {
                        panic!("Tile::scale: not writable")
                    }
                }
            }
            ComptimeOption::None => {}
        }
    }

    /// The fragment grid this accumulator holds and one fragment's `m × n`: a partition's own
    /// grid, a single plane tile's `1 × 1` of its whole window.
    pub(crate) fn fragment_grid(&self) -> comptime_type!(((usize, usize), usize, usize)) {
        let edges = comptime!(MatrixAxes::edges(&self.place.space));
        let rows = comptime!(self.place.space.extent_at(edges.row_split));
        let cols = comptime!(self.place.space.extent_at(edges.col_split));
        match &self.kind {
            TileKind::PlanePartition(p) => {
                comptime!(((p.m_tiles, p.n_tiles), rows / p.m_tiles, cols / p.n_tiles))
            }
            TileKind::PlaneTile(_) => comptime!(((1, 1), rows, cols)),
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lanes(_) => {
                panic!("Tile::fragment_grid: only a plane-resident accumulator holds fragments")
            }
        }
    }

    /// Initialize this tile with `val`. Same shape as [`zero`](Tile::zero).
    pub fn init(&mut self, val: T) {
        match &mut self.kind {
            TileKind::Memory(d) => d.init(val),
            TileKind::PlaneTile(t) => t.init(val),
            TileKind::PlanePartition(p) => p.init(val),
            TileKind::TmaGmem(_) => panic!("Tile::init: a tma source is not writable"),
            TileKind::Procedural(_) | TileKind::Lanes(_) => {
                panic!("Tile::init: a procedural tile and the plane's lanes are not writable")
            }
        }
    }

    /// The window as one dense run of `Vector<T, W>` lines (`W` the store's own width): index `i`
    /// reads line `origin + i`, one add and no layout walk. See [`Memory::dense_lines`] for the
    /// caller-owned contiguity contract; the streaming fold's operands satisfy it by construction.
    pub fn dense<W: Size>(&self) -> &[Vector<T, W>] {
        self.mem("dense").dense_lines::<W>()
    }

    /// The mutable twin of [`dense`](Tile::dense).
    pub(crate) fn dense_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        self.mem_mut("dense_mut").dense_lines_mut::<W>()
    }

    /// Move `src` into `self`, the physical pairing picking the instruction that does it. A
    /// partition source is matched first because it needs the whole destination tile, which the
    /// pairing match below would keep borrowed.
    pub fn copy_from(&mut self, src: &Tile<T>) {
        // Bound before the match, which borrows the kind: a memory fill needs the logical space
        // both sides carry (a gathered source is addressed per axis).
        let space = comptime!(self.place.space.clone());
        match &src.kind {
            // One fragment is stored as the fragment; a grid of them is stored one at a time,
            // by the loop over its cells the kernel writes.
            TileKind::PlanePartition(s) => match &mut self.kind {
                TileKind::Memory(d) => s.fragment().store_window(d, space),
                TileKind::PlaneTile(_)
                | TileKind::PlanePartition(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lanes(_) => {
                    panic!("Tile::copy_from: a fragment stores into memory")
                }
            },
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lanes(_) => match (&mut self.kind, &src.kind) {
                // The lanes are filled from the memory window of the box they were opened over.
                (TileKind::Lanes(d), TileKind::Memory(_)) => d.load(src),
                (TileKind::PlanePartition(d), TileKind::Memory(_)) => d.fill_from(src),
                (TileKind::PlaneTile(d), TileKind::Memory(_)) => d.load_window(src),
                (TileKind::Memory(d), TileKind::PlaneTile(s)) => s.store_window(d, space),
                (TileKind::Memory(d), TileKind::TmaGmem(s)) => s.load_into(d),
                (TileKind::Memory(d), TileKind::Memory(s)) => d.fill_from(s, space),
                (TileKind::Memory(d), TileKind::Procedural(s)) => d.fill_procedural(s, space),
                (TileKind::PlaneTile(_), TileKind::PlaneTile(_)) => {
                    panic!("Tile::copy_from: plane tile to plane tile cast not wired")
                }
                _ => panic!("Tile::copy_from: unsupported kind pairing"),
            },
        }
    }

    /// Add a resident block `src` into this one, casting up to `T`: the **promotion** a leaf
    /// accumulating in a narrower element drains through, register to register, without
    /// touching memory.
    ///
    /// The resident counterpart of [`copy_cast_from`](Self::copy_cast_from), which drains a block
    /// *out* to its sink. It lets a leaf contract in the operands' own element while the sum across
    /// leaf calls stays wide, bounding the error by the leaf's own depth.
    ///
    /// On a target with no matrix units an `f16` block reaches `hfma2`, twice the arithmetic of an
    /// `f32` one, but `f16` counts integers exactly only to 2048, so a contraction accumulated in
    /// it end to end stops advancing part way through.
    ///
    /// Both sides are register blocks over one sink, so the add is line for line
    /// ([`RegisterData::add_cast_from`]). A partition promotes cell by cell.
    pub fn add_cast_from<S: Numeric>(&mut self, src: &Tile<S>) {
        match (&mut self.kind, &src.kind) {
            (TileKind::PlaneTile(d), TileKind::PlaneTile(s)) => d.add_cast_from(s),
            (TileKind::PlanePartition(d), TileKind::PlanePartition(s)) => {
                // The handle is the storage: a partition's fragment clones the block's array
                // reference, so promoting through the clone promotes the block itself.
                let mut fragment = d.fragment();
                fragment.add_cast_from(&s.fragment())
            }
            _ => panic!(
                "Tile::add_cast_from: a resident block promotes into a resident block; memory \
                 is reached through `copy_cast_from`"
            ),
        }
    }

    /// [`copy_from`](Self::copy_from) with a cast: a resident fragment `src`, wider than this
    /// memory window, stored down to `T`. How an accumulator's cells reach an output of the
    /// output's own type, one fragment per call, from the loop the kernel writes over its cells.
    ///
    /// Into a destination that folds ([`Write::Accumulate`]), a cmma fragment drains through the
    /// scratch its accumulator was opened with, cell by cell, since its intrinsic's store cannot
    /// add.
    pub fn copy_cast_from<S: Numeric>(&mut self, src: &Tile<S>) {
        let space = comptime!(self.place.space.clone());
        match (&mut self.kind, &src.kind) {
            (TileKind::Memory(d), TileKind::PlaneTile(s)) => s.store_cast_window(d, space),
            (TileKind::Memory(d), TileKind::PlanePartition(s)) => {
                s.fragment().store_cast_window(d, space)
            }
            _ => panic!("Tile::copy_cast_from: a fragment stores into memory; nothing else casts"),
        }
    }

    /// Spill this plane-resident tile into its slot of the plane's scratch, the first half of a
    /// bounce. [`drained_into`](Self::drained_into) owns the barriers around it.
    pub fn spill_to_scratch(&self) {
        match &self.kind {
            TileKind::PlaneTile(t) => t.spill_to_scratch(),
            TileKind::PlanePartition(p) => p.fragment().spill_to_scratch(),
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lanes(_) => {
                panic!("Tile::spill_to_scratch: a plane-resident tile spills; nothing else does")
            }
        }
    }

    /// Add `src`'s spilled cells into this memory window, the second half of a bounce. Where this
    /// store folds, the add is its atomic one.
    pub fn add_from_scratch<S: Numeric>(&mut self, src: &Tile<S>) {
        let space = comptime!(self.place.space.clone());
        match (&mut self.kind, &src.kind) {
            (TileKind::Memory(d), TileKind::PlaneTile(s)) => s.add_from_scratch(d, space),
            (TileKind::Memory(d), TileKind::PlanePartition(s)) => {
                s.fragment().add_from_scratch(d, space)
            }
            // A tuple match, which the cube macro reads as plain Rust, so the wildcard is allowed
            // here where a single-value match on the enum would need every arm.
            _ => panic!("Tile::add_from_scratch: a spilled plane tile adds into memory"),
        }
    }

    /// Drain this plane-resident accumulator into `dest`, cast to `dest`'s element, over the tiles
    /// the `cells` level names — `None` where the accumulator is one tile and the drain one store.
    ///
    /// **`self` and `dest` are indexed by the same region**, so a caller that narrows one narrows
    /// the other to the same window. An accumulator opened wider than the destination handed over
    /// resolves each region somewhere else and drains the wrong cells.
    ///
    /// **The scratch's size decides the barrier count** ([`Resident`]). With one tile resident
    /// each tile drains on its own, three cube-wide barriers apiece. With the whole partition
    /// resident no two tiles share a slot: every spill, one barrier, every add, two in all.
    pub fn drained_into<Out: Numeric>(&self, dest: &Tile<Out>, #[comptime] cells: Option<Level>) {
        match cells {
            // A grid below the drain: one region of `cells` per tile of it.
            Some(cells) => self.drain_grid(dest, cells),
            // No grid: the accumulator is one tile and the drain is one store.
            // The register leaves are this shape, since their block is the
            // plane's whole box rather than a partition of it.
            None => self.drain_tile(dest),
        }
    }

    /// [`drained_into`](Self::drained_into) over a grid of tiles.
    fn drain_grid<Out: Numeric>(&self, dest: &Tile<Out>, #[comptime] cells: Level) {
        let resident = self.resident();
        if comptime!(!resident.bounces()) {
            for region in dest.over(&cells).unrolled() {
                let mut window = dest.at(&region);
                window.copy_cast_from(&self.at(&region));
            }
        } else if comptime!(resident.drains_together()) {
            // Every tile has its own slot, so every spill can happen before any add.
            sync_cube();
            for region in dest.over(&cells).unrolled() {
                self.at(&region).spill_to_scratch();
            }
            sync_cube();
            for region in dest.over(&cells).unrolled() {
                let mut window = dest.at(&region);
                window.add_from_scratch(&self.at(&region));
            }
            sync_cube();
        } else {
            // One slot between them, so each tile's spill and add pair off inside the loop.
            for region in dest.over(&cells).unrolled() {
                let mut window = dest.at(&region);
                sync_cube();
                self.at(&region).spill_to_scratch();
                sync_cube();
                window.add_from_scratch(&self.at(&region));
                sync_cube();
            }
        }
    }

    /// [`drained_into`](Self::drained_into) for a single tile.
    fn drain_tile<Out: Numeric>(&self, dest: &Tile<Out>) {
        let resident = self.resident();
        let mut window = dest.clone();
        if comptime!(!resident.bounces()) {
            window.copy_cast_from(self);
        } else {
            sync_cube();
            self.spill_to_scratch();
            sync_cube();
            window.add_from_scratch(self);
            sync_cube();
        }
    }

    /// How much of this accumulator its scratch holds, which is what tells a drain whether to
    /// bounce at all and whether it may hoist its barriers.
    pub(crate) fn resident(&self) -> comptime_type!(Resident) {
        match &self.kind {
            TileKind::PlanePartition(p) => comptime!(p.resident),
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lanes(_) => comptime!(Resident::None),
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
        match &self.kind {
            // A staged operand answers against the window it was *filled from*: the fill wrote the
            // boundary's value wherever a tap fell outside, which this window no longer records. A
            // stage with no source window was never gathered, so every tap is in bounds; a global
            // buffer answers against its own window.
            TileKind::Memory(data) => {
                let projection = comptime!(data.projection.clone());
                if comptime!(!projection.addresses(axis)) {
                    true.runtime()
                } else {
                    #[comptime]
                    match &data.source_window {
                        ComptimeOption::Some(source) => source.axes_in_bounds(
                            &data.window.origin,
                            pos,
                            comptime!(projection.carriers(axis).to_vec()),
                        ),
                        ComptimeOption::None => match comptime!(data.address) {
                            AddressSpace::Global => data
                                .window
                                .axes_in_bounds(pos, comptime!(projection.carriers(axis).to_vec())),
                            AddressSpace::Shared => true.runtime(),
                        },
                    }
                }
            }
            TileKind::Procedural(data) => data.axis_in_bounds(pos, axis),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lanes(_) => {
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
        factors.map(|factors| match &self.kind {
            TileKindExpand::Procedural(data) => (0..factors)
                .map(|f| {
                    (
                        data.__expand_factor_reads_axis_method(scope, f, row),
                        data.__expand_factor_reads_axis_method(scope, f, col),
                    )
                })
                .collect(),
            TileKindExpand::Memory(_)
            | TileKindExpand::PlaneTile(_)
            | TileKindExpand::PlanePartition(_)
            | TileKindExpand::TmaGmem(_)
            | TileKindExpand::Lanes(_) => (0..factors).map(|_| (true, true)).collect(),
        })
    }
}

/// `space` with each [`Dynamic`](crate::Extent) axis sized by the first of `a`, `b`, `c` that
/// [`witnesses`](Tile::witnesses) it: the runtime space an operation's loops walk. A fully-`Static`
/// space short-circuits. One tile may stand for all three ([`runtime_space`](Tile::runtime_space)).
#[cube]
pub(crate) fn witnessed_space<A: Numeric, B: Numeric, C: Numeric>(
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
            // `sizes` is positional, so every axis pushes, but [`Extents::count`] folds a `Static`
            // axis to its comptime extent. Fold it here too rather than asking an operand: one
            // `Dynamic` axis must not make the `Static` ones unreadable on a tile with no bound.
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

/// Where a tile sits in its partitioning: its space, and the levels below its depth. The same
/// read on a tile and on the tile as comptime code sees it, so a comptime derivation
/// ([`Fragments::below`](crate::Fragments::below)) takes either.
pub trait Placed {
    fn space(&self) -> &Space;
    fn below(&self) -> &[Level];
}

impl<T: Numeric> Placed for Tile<T> {
    fn space(&self) -> &Space {
        &self.place.space
    }
    fn below(&self) -> &[Level] {
        self.place.below()
    }
}

impl<T: Numeric> Placed for TileExpand<T> {
    fn space(&self) -> &Space {
        &self.place.space
    }
    fn below(&self) -> &[Level] {
        self.place.below()
    }
}

/// The runtime twin of `for plane in tile`, which a kernel's host-side body names but never
/// runs: every loop over a tile expands in-kernel.
impl<T: Numeric> IntoIterator for Tile<T> {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        unexpanded!()
    }
}

impl<T: Numeric> IntoIterator for &Tile<T> {
    type Item = Region;
    type IntoIter = std::vec::IntoIter<Region>;

    fn into_iter(self) -> Self::IntoIter {
        unexpanded!()
    }
}

/// `for plane in tile` iterates the level below the tile over its own box.
impl<T: Numeric> Iterable for TileExpand<T> {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        let walk = self.__expand_walk_method(scope);
        if walk.const_len() == Some(1) {
            walk.expand_unroll(scope, body)
        } else {
            walk.expand(scope, body)
        }
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        self.__expand_walk_method(scope).expand_unroll(scope, body)
    }
}

/// `for plane in &tile`: the same, leaving `tile` to the body.
impl<T: Numeric> Iterable for &TileExpand<T> {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        self.clone().expand(scope, body)
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        self.clone().expand_unroll(scope, body)
    }
}
