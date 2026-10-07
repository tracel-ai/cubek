//! The [`Tile`]: one operand's backing store plus the comptime [`Space`] it projects.

use cubecl::{prelude::*, std::tensor::layout::CoordsDyn, unexpanded};

use cubecl::zspace::SmallVec;

use crate::stage::pipeline::payload::base::{StageSpec, stage_one};
use crate::*;

/// One operand's data: a runtime backing store and the comptime [`Space`] it projects.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Tile<T: Numeric> {
    pub(crate) kind: TileKind<T>,
    /// Where this tile sits in its partitioning.
    #[cube(comptime)]
    pub(crate) place: Placement,
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// Pairs `kind` with its `place`.
    pub(crate) fn new(kind: TileKind<T>, #[comptime] place: Placement) -> Tile<T> {
        Tile::<T> { kind, place }
    }

    /// This tile's own box at this depth.
    pub fn space(&self) -> comptime_type!(Space) {
        comptime!(self.place.space.clone())
    }

    /// These cells placed at `depth` of `levels`: a tile whose grid the caller states.
    pub fn with_levels(&self, #[comptime] depth: usize, #[comptime] levels: Vec<Level>) -> Tile<T> {
        Tile::new(
            self.kind.clone(),
            comptime!(Placement::new(self.place.space.clone(), depth, levels)),
        )
    }

    /// This tile's memory store; panics for kinds with none, naming `site`.
    pub(crate) fn mem(&self, #[comptime] site: &str) -> &Memory<T> {
        match &self.kind {
            TileKind::Memory(g) => g,
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::{site}: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::{site}: a tma source has no element view"),
            TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!("Tile::{site}: a procedural tile and the plane's units have no memory view")
            }
        }
    }

    /// The recipe this procedural tile evaluates; panics for every other kind.
    pub(crate) fn recipe(&self) -> &Procedural<T> {
        match &self.kind {
            TileKind::Procedural(recipe) => recipe,
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => panic!("Tile::recipe: only a procedural tile has a recipe"),
        }
    }

    pub(crate) fn mem_mut(&mut self, #[comptime] site: &str) -> &mut Memory<T> {
        match &mut self.kind {
            TileKind::Memory(g) => g,
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::{site}: a plane tile has no memory view")
            }
            TileKind::TmaGmem(_) => panic!("Tile::{site}: a tma source is not writable"),
            TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!("Tile::{site}: a procedural tile and the plane's units are not writable")
            }
        }
    }

    /// Who moves this operand's bytes into a stage. Panics on a plane fragment.
    pub fn delivery(&self) -> comptime_type!(Delivery) {
        match &self.kind {
            TileKind::Memory(d) => comptime!(d.access.delivery),
            TileKind::TmaGmem(_) => comptime!(Delivery::Tma),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::delivery: a resident fragment is not a stage source")
            }
            TileKind::Procedural(_) | TileKind::Lines(_) => comptime!(Delivery::Procedural),
        }
    }

    /// Whether a cmma fragment draining into this tile goes through a scratch
    /// ([`FragmentDrain::Bounce`]): a destination that adds, overhangs, or has no address.
    pub(crate) fn fragments_bounce(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(m) => {
                let addressed = m.store.addressed();
                comptime!(FragmentDrain::of(&m.access, addressed) == FragmentDrain::Bounce)
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => comptime!(false),
        }
    }

    pub(crate) fn runtime_map(&self) -> RuntimeMap {
        match &self.kind {
            TileKind::Memory(g) => g.map.clone(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => RuntimeMap::integral(comptime!(self.place.space.rank())),
        }
    }

    /// The launch's cube size this operand was bound with, `0` when unknown.
    pub(crate) fn units(&self) -> comptime_type!(usize) {
        match &self.kind {
            TileKind::Memory(d) => comptime!(d.access.fill.count),
            TileKind::TmaGmem(t) => comptime!(t.units),
            TileKind::Procedural(_)
            | TileKind::Lines(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_) => {
                comptime!(0)
            }
        }
    }

    /// The units that share a cooperative fill of this tile ([`FillUnits`]): one plane's for a
    /// stage that plane owns, the cube's for everything else.
    pub(crate) fn fill_units(&self) -> comptime_type!(FillUnits) {
        match &self.kind {
            TileKind::Memory(d) => comptime!(d.access.fill),
            TileKind::TmaGmem(t) => comptime!(FillUnits::cube(t.units)),
            TileKind::Procedural(_)
            | TileKind::Lines(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_) => comptime!(FillUnits::cube(0)),
        }
    }

    /// Physical vector width of the backing store; `1` for a fragment or tma source. Refuses a
    /// memory whose loads span several axes: its reader asks for the vector tile instead.
    pub fn vector_size(&self) -> comptime_type!(usize) {
        match &self.kind {
            TileKind::Memory(d) => {
                let load = d.vector_tile(&comptime!(self.place.space.clone()));
                comptime!(load.along_one_axis("a reader asking Tile::vector_size").1)
            }
            TileKind::Lines(c) => c.line(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(1usize)
            }
        }
    }

    /// What one vector load of this tile covers: a run along the innermost axis, or the stored
    /// tiles that hold [`vector_size`](Tile::vector_size) values ([`VectorTile::new`]).
    pub(crate) fn vector_tile(&self) -> comptime_type!(VectorTile) {
        let space = comptime!(self.place.space.clone());
        match &self.kind {
            TileKind::Memory(d) => d.vector_tile(&space),
            TileKind::Lines(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                let width = self.vector_size();
                comptime!(VectorTile::run(space.axis_at(space.rank() - 1), width))
            }
        }
    }

    /// Ask this accumulator to start from `init_from`; returns what took. A folding destination
    /// always answers [`Identity`](InitFrom::Identity).
    pub(crate) fn request_init_from(
        &mut self,
        #[comptime] init_from: InitFrom,
    ) -> comptime_type!(InitFrom) {
        match &mut self.kind {
            TileKind::Memory(d) => {
                let init_from = comptime!(match d.access.write {
                    Write::Accumulate | Write::Exclusive(_) => InitFrom::Identity,
                    Write::Replace => init_from,
                });
                d.set_init_from(comptime!(init_from));
                comptime!(init_from)
            }
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                comptime!(InitFrom::Cell)
            }
        }
    }

    /// What a stage of this operand holds ([`StageElement`]).
    pub(crate) fn stage_element(&self) -> comptime_type!(StageElement) {
        match &self.kind {
            TileKind::Memory(d) => d.stage_element(),
            TileKind::TmaGmem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                comptime!(StageElement::Served)
            }
        }
    }

    /// How this tile's values sit in memory; see [`Memory::packing`].
    pub(crate) fn packing(&self) -> comptime_type!(Packing) {
        match &self.kind {
            TileKind::Memory(d) => d.packing(),
            TileKind::Lines(c) => c.packing(),
            TileKind::TmaGmem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::Procedural(_) => {
                comptime!(Packing::Plain)
            }
        }
    }

    /// Whether this tile's mapping has overlapping windows, so only an N-D view describes it.
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
            | TileKind::Lines(_) => comptime!(false),
        }
    }

    /// Whether this tile is a memory window.
    pub(crate) fn is_memory(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(_) => comptime!(true),
            TileKind::Procedural(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => comptime!(false),
        }
    }

    /// Whether this tile is backed by shared memory, the only space a `sync_cube()` orders.
    pub fn is_shared(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(m) => comptime!(m.address == AddressSpace::Shared),
            TileKind::Procedural(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => comptime!(false),
        }
    }

    /// `Some(n)` for a recipe stating `n` separable factors, `None` otherwise.
    pub(crate) fn factors(&self) -> comptime_type!(Option<usize>) {
        match &self.kind {
            TileKind::Procedural(data) => data.factorization(),
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => comptime!(None),
        }
    }

    pub(crate) fn window_signed(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(d) => comptime!(d.window.signed),
            _ => comptime!(false),
        }
    }

    /// This operand's per-axis boundary handling, which a stage filled from it inherits.
    pub(crate) fn window_boundaries(
        &self,
    ) -> comptime_type!(SmallVec<[Option<Boundary>; Space::MAX_RANK]>) {
        match &self.kind {
            TileKind::Memory(d) => comptime!(d.window.boundaries.clone()),
            _ => comptime!(SmallVec::new()),
        }
    }

    /// The boundaries factor-local normalization tests, kept on a stage's source window.
    pub(crate) fn separable_boundaries(
        &self,
    ) -> comptime_type!(SmallVec<[Option<Boundary>; Space::MAX_RANK]>) {
        match &self.kind {
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
            | TileKind::Lines(_) => comptime!(SmallVec::new()),
        }
    }

    /// The separable factor-normalization request of a procedural tile, if any.
    pub(crate) fn factor_normalization(&self) -> comptime_type!(Option<Normalization>) {
        match &self.kind {
            TileKind::Procedural(data) => comptime!(data.normalization.clone()),
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => comptime!(None),
        }
    }

    /// One factor of a separable recipe at `pos`; requires [`factors`](Tile::factors) to be `Some`.
    pub(crate) fn separable_factor(&self, pos: CoordsDyn, #[comptime] factor: usize) -> T {
        match &self.kind {
            TileKind::Procedural(data) => {
                data.read_factor_at(&pos, factor, comptime!(self.place.space.clone()))
            }
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Lines(_) => {
                panic!(
                    "Tile::separable_factor: a tile read from a buffer states no factorization, \
                     so `factors` answered `None` and there is no factor {factor} to evaluate"
                )
            }
        }
    }

    /// Whether this tile can state `axis`'s runtime size.
    pub(crate) fn witnesses(&self, #[comptime] axis: Axis) -> comptime_type!(bool) {
        let bounded = self.bounded();
        let projection = self.projection();
        comptime!(
            bounded
                && self.place.space.contains(axis)
                && self.place.space.is_dynamic(axis)
                && Witness::new(&projection, axis).is_some()
        )
    }

    /// How this tile's logical axes address its buffer's physical ones.
    pub(crate) fn projection(&self) -> comptime_type!(Projection) {
        match &self.kind {
            TileKind::Memory(g) => comptime!(g.projection.clone()),
            TileKind::Lines(c) => c.projection(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(Projection::direct_over(&self.place.space))
            }
        }
    }

    /// This tile at `like`'s place in its nest.
    pub(crate) fn nested_like<U: Numeric>(self, like: &Tile<U>) -> Tile<T> {
        Tile::<T> {
            kind: self.kind,
            place: comptime!(Placement::new(
                self.place.space.clone(),
                like.place.depth,
                like.place.levels.clone()
            )),
        }
    }

    /// This operand's window at `from` on `axis`, reading zero past `until` (both in elements).
    pub fn within(&self, #[comptime] axis: Axis, from: usize, until: usize) -> Tile<T> {
        let tile_kind = match &self.kind {
            TileKind::Memory(g) => TileKind::new_Memory(g.within(axis, from, until)),
            TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_) => panic!(
                "Tile::within: only a memory operand carries a window to place; a fragment, a \
                 tensor-map source and a recipe have none"
            ),
        };
        Tile::new(tile_kind, comptime!(self.place.clone()))
    }

    /// Window this tile down to `region`, no copy.
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

    fn at_from(&self, region: &Region, #[comptime] i: usize) -> Tile<T> {
        let sub = self.at_step(&region.step(i));
        if comptime!(i + 1 == region.path.len()) {
            sub
        } else {
            sub.at_from(region, comptime!(i + 1))
        }
    }

    /// One level down: window this tile to `step`'s box.
    pub(crate) fn at_step(&self, step: &Step) -> Tile<T> {
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
            TileKind::Lines(c) => {
                TileKind::new_Lines(c.at(step, comptime!(self.place.space.clone())))
            }
            // A plane tile passes through; legal only where the level cuts nothing on m/n.
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

    /// Whether this tile has a buffer bound to read an extent off.
    fn bounded(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Memory(_) | TileKind::TmaGmem(_) => comptime!(true),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                comptime!(false)
            }
        }
    }

    /// This operand's runtime logical size along `axis`; only for an axis it
    /// [`witnesses`](Tile::witnesses).
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
            TileKind::Memory(g) => g.window.bound.at(p).cast::<usize>(),
            TileKind::TmaGmem(t) => t.bound[p].cast::<usize>(),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::runtime_extent: a plane tile has no extent")
            }
            TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!(
                    "Tile::runtime_extent: a procedural tile and the plane's units have no extent"
                )
            }
        };
        // `bound` counts lines on the vectorized innermost axis.
        let last = comptime!(projection.physical_rank() - 1);
        let w = self.vector_size();
        comptime!(if p == last { w } else { 1usize }) * raw
    }

    /// This tile's own box as a runtime space, dynamic axes sized off its buffer.
    pub(crate) fn runtime_space(&self) -> Space {
        witnessed_space(comptime!(self.place.space.clone()), self, self, self)
    }

    /// The regions of the level below this tile over its own box.
    pub fn walk(&self) -> Walk {
        let space = self.runtime_space();
        Region::rooted(
            &space,
            comptime!(self.place.levels.clone()),
            comptime!(self.place.depth),
        )
        .walk()
    }

    /// The regions of `level` over this tile's own box.
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

    /// Zero this whole tile.
    pub fn zero(&mut self) {
        match &mut self.kind {
            TileKind::Memory(d) => d.zero(),
            TileKind::PlaneTile(t) => t.zero(),
            TileKind::PlanePartition(p) => p.zero(),
            TileKind::TmaGmem(_) => panic!("Tile::zero: a tma source is not writable"),
            TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!("Tile::zero: a procedural tile and the plane's units are not writable")
            }
        }
    }

    /// Seed this tile with `monoid`'s identity.
    pub(crate) fn init_identity(&mut self, #[comptime] monoid: Monoid) {
        match comptime!(monoid) {
            Monoid::Sum => self.zero(),
            Monoid::Prod | Monoid::Max | Monoid::Min => self.init(Monoid::identity::<T>(monoid)),
        }
    }

    /// The one value this tile holds.
    pub(crate) fn only(&self) -> T {
        let axes = comptime!(MatrixAxes::trailing(&self.place.space));
        let matrix = self.matrix_packed::<Const<1>>(axes, 0usize);
        let origin = 0u32.runtime();
        matrix.read((origin, origin)).extract(0usize)
    }

    /// Multiply every partial this accumulator holds by `factor`'s one value, if bound.
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
                    TileKind::TmaGmem(_) | TileKind::Procedural(_) | TileKind::Lines(_) => {
                        panic!("Tile::scale: not writable")
                    }
                }
            }
            ComptimeOption::None => {}
        }
    }

    /// The fragment grid this accumulator holds and one fragment's `m × n`.
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
            | TileKind::Lines(_) => {
                panic!("Tile::fragment_grid: only a plane-resident accumulator holds fragments")
            }
        }
    }

    /// Initialize this tile with `val`.
    pub fn init(&mut self, val: T) {
        match &mut self.kind {
            TileKind::Memory(d) => d.init(val),
            TileKind::PlaneTile(t) => t.init(val),
            TileKind::PlanePartition(p) => p.init(val),
            TileKind::TmaGmem(_) => panic!("Tile::init: a tma source is not writable"),
            TileKind::Procedural(_) | TileKind::Lines(_) => {
                panic!("Tile::init: a procedural tile and the plane's units are not writable")
            }
        }
    }

    /// The window as one dense run of `Vector<T, W>` lines; see `Memory::dense_lines` for the
    /// contiguity contract.
    pub fn dense<W: Size>(&self) -> &[Vector<T, W>] {
        self.mem("dense").dense_lines::<W>()
    }

    /// The mutable twin of [`dense`](Tile::dense).
    pub fn dense_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        self.mem_mut("dense_mut").dense_lines_mut::<W>()
    }

    /// A fresh tile shaped to stage one region of `level` of this operand, laid out as `storage`.
    pub fn stage(&self, #[comptime] level: Level, #[comptime] storage: StageStorage) -> Tile<T> {
        Memory::<T>::stage(
            self,
            level,
            storage,
            comptime!(None),
            comptime!(StageOwner::Cube),
        )
    }

    /// A fresh stage of this operand for one region of `walk`, laid out as `storage`: shaped,
    /// owned and placed as the walk's stages are, the cube's or each plane's own copy. What a
    /// kernel stages once before a walk that leaves the operand unchanged, as an attention's
    /// query beside the keys it walks.
    pub fn stage_for(&self, walk: &Walk, #[comptime] storage: StageStorage) -> Tile<T> {
        let owner = walk.stage_owner();
        stage_one(
            self,
            comptime!(StageSpec {
                level: walk.level.clone(),
                depth: walk.depth(),
                storage,
                width: None,
                owner,
            }),
        )
    }

    /// A fresh shared-memory tile over `axes` of one region of `walk`, laid out as `storage` and
    /// served one value a line: placed where the walk's regions sit, and owned as the walk's
    /// stages are, one for the cube or a copy for each plane ([`Stages`]).
    ///
    /// What a kernel holds between two steps of its walk that no operand is: an attention's
    /// scores between its two contractions. Placed in the partitioning, it opens an accumulator
    /// and is windowed by the walk's regions like an operand.
    pub fn scratch(
        walk: &Walk,
        #[comptime] axes: Vec<Axis>,
        #[comptime] storage: StageStorage,
    ) -> Tile<T> {
        let owner = walk.stage_owner();
        let place = comptime!(Placement::new(
            walk.level.child(&walk.space).subspace(&axes),
            walk.depth(),
            walk.parent.path.root_levels(),
        ));
        let tile =
            Memory::<T>::smem_owned(place.space.clone(), 1usize, storage, 0usize, 0usize, owner);
        Tile::new(tile.kind, place)
    }

    /// A fresh shared-memory tile over `space`, laid out as `storage`, serving one value a line.
    pub fn shared(#[comptime] space: Space, #[comptime] storage: StageStorage) -> Tile<T> {
        Memory::<T>::smem(space, comptime!(1usize), storage, comptime!(0usize))
    }

    /// A fresh shared-memory tile over `space`, served `vector_size` wide and filled by `units`
    /// units (`0`: the whole cube).
    pub fn smem(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] storage: StageStorage,
        #[comptime] units: usize,
    ) -> Tile<T> {
        Memory::<T>::smem(space, vector_size, storage, units)
    }

    /// Move `src` into `self`, the kind pairing picking the instruction.
    pub fn copy_from(&mut self, src: &Tile<T>) {
        let (scaled, into_memory) = (src.scaled(), self.is_memory());
        comptime!(assert!(
            !scaled || into_memory,
            "Tile::copy_from: a scaled source decodes into memory only; to load it into \
             fragments, contract it: the fragment lands it"
        ));
        if comptime!(scaled) {
            self.copy_scaled_from(src);
        } else {
            self.copy_unscaled_from(src);
        }
    }

    fn copy_unscaled_from(&mut self, src: &Tile<T>) {
        // Bound before the match, which borrows the kind.
        let space = comptime!(self.place.space.clone());
        match &src.kind {
            TileKind::PlanePartition(s) => match &mut self.kind {
                TileKind::Memory(d) => s.fragment().store_window(d, space),
                TileKind::PlaneTile(_)
                | TileKind::PlanePartition(_)
                | TileKind::TmaGmem(_)
                | TileKind::Procedural(_)
                | TileKind::Lines(_) => {
                    panic!("Tile::copy_from: a fragment stores into memory")
                }
            },
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => match (&mut self.kind, &src.kind) {
                (TileKind::Lines(d), TileKind::Memory(_)) => d.load(src),
                (TileKind::PlanePartition(d), TileKind::Memory(_)) => d.fill_from(src),
                (TileKind::PlaneTile(d), TileKind::Memory(_)) => d.load_window(src),
                (TileKind::Memory(d), TileKind::PlaneTile(s)) => s.store_window(d, space),
                (TileKind::Memory(d), TileKind::TmaGmem(s)) => s.load_into(d),
                (TileKind::Memory(d), TileKind::Memory(s)) => d.load_from(s, space),
                (TileKind::Memory(d), TileKind::Procedural(s)) => d.fill_procedural(s, space),
                (TileKind::PlaneTile(_), TileKind::PlaneTile(_)) => {
                    panic!("Tile::copy_from: plane tile to plane tile cast not wired")
                }
                _ => panic!("Tile::copy_from: unsupported kind pairing"),
            },
        }
    }

    /// Add a resident block `src` into this one, casting up to `T`, without touching memory.
    pub fn add_cast_from<S: Numeric>(&mut self, src: &Tile<S>) {
        match (&mut self.kind, &src.kind) {
            (TileKind::PlaneTile(d), TileKind::PlaneTile(s)) => d.add_cast_from(s),
            (TileKind::PlanePartition(d), TileKind::PlanePartition(s)) => {
                // The fragment clones the block's array reference, so this promotes the block itself.
                let mut fragment = d.fragment();
                fragment.add_cast_from(&s.fragment())
            }
            _ => panic!(
                "Tile::add_cast_from: a resident block promotes into a resident block; memory \
                 is reached through `copy_cast_from`"
            ),
        }
    }

    /// [`copy_from`](Tile::copy_from) with a cast: stores a wider resident fragment down to `T`,
    /// or copies a memory window of the same box, the units sharing it a line at a time.
    pub fn copy_cast_from<S: Numeric>(&mut self, src: &Tile<S>) {
        let space = comptime!(self.place.space.clone());
        let holder = comptime!(self.place.holder());
        match (&mut self.kind, &src.kind) {
            (TileKind::Memory(d), TileKind::PlaneTile(s)) => s.store_cast_window(d, space),
            (TileKind::Memory(d), TileKind::PlanePartition(s)) => {
                s.fragment().store_cast_window(d, space)
            }
            // A window its holder shares is filled by the holder's units, whoever filled the
            // buffer it windows: a plane's window of the cube's scratch is the plane's.
            (TileKind::Memory(d), TileKind::Memory(s)) => {
                d.filled_by(holder).fill_cast_from(s, space)
            }
            _ => panic!(
                "Tile::copy_cast_from: a fragment or a memory window stores into memory; nothing \
                 else casts"
            ),
        }
    }

    /// Spill this plane-resident tile into its slot of the plane's scratch.
    pub(crate) fn spill_to_scratch(&self) {
        match &self.kind {
            TileKind::PlaneTile(t) => t.spill_to_scratch(),
            TileKind::PlanePartition(p) => p.fragment().spill_to_scratch(),
            TileKind::Memory(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => {
                panic!("Tile::spill_to_scratch: a plane-resident tile spills; nothing else does")
            }
        }
    }

    /// Add `src`'s spilled cells into this memory window.
    pub(crate) fn add_from_scratch<S: Numeric>(&mut self, src: &Tile<S>) {
        let space = comptime!(self.place.space.clone());
        match (&mut self.kind, &src.kind) {
            (TileKind::Memory(d), TileKind::PlaneTile(s)) => s.add_from_scratch(d, space),
            (TileKind::Memory(d), TileKind::PlanePartition(s)) => {
                s.fragment().add_from_scratch(d, space)
            }
            _ => panic!("Tile::add_from_scratch: a spilled plane tile adds into memory"),
        }
    }

    /// Whether one factor tap lands inside the input axes that factor moves.
    pub(crate) fn separable_physical_tap_in_bounds(
        &self,
        pos: &CoordsDyn,
        #[comptime] axis: Axis,
    ) -> bool {
        match &self.kind {
            // A stage answers against the window it was filled from.
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
            | TileKind::Lines(_) => {
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
    /// This tile's own box, readable inside `comptime!`.
    pub fn space(&self) -> Space {
        self.place.space.clone()
    }

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
                        data.__expand_factor_reads_method(scope, f, row),
                        data.__expand_factor_reads_method(scope, f, col),
                    )
                })
                .collect(),
            TileKindExpand::Memory(_)
            | TileKindExpand::PlaneTile(_)
            | TileKindExpand::PlanePartition(_)
            | TileKindExpand::TmaGmem(_)
            | TileKindExpand::Lines(_) => (0..factors).map(|_| (true, true)).collect(),
        })
    }
}

/// `space` with each [`Dynamic`](crate::Extent) axis sized by the first of `a`, `b`, `c` that
/// [`witnesses`](Tile::witnesses) it.
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
            // Fold `Static` axes here so one `Dynamic` axis doesn't require a bound for the rest.
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

/// Host-side stub of `for plane in tile`; never runs.
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

/// `for plane in &tile`.
impl<T: Numeric> Iterable for &TileExpand<T> {
    type Item = RegionExpand;

    fn expand(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        self.clone().expand(scope, body)
    }

    fn expand_unroll(self, scope: &Scope, body: &mut dyn FnMut(&Scope, RegionExpand)) {
        self.clone().expand_unroll(scope, body)
    }
}

impl<E: Numeric> Tile<E> {
    /// These values under `scale`, multiplied in where they are read. Every axis of `scale` must
    /// be one of the values'.
    pub fn mul<S: Numeric>(&self, _scale: &Tile<S>) -> Tile<E> {
        unexpanded!()
    }

    /// The factor a leaf reads these values under.
    pub(crate) fn reader(
        &self,
        _axes: MatrixAxes,
        _matrix: usize,
        _side: Side,
        _out: Space,
        _acc_axes: MatrixAxes,
    ) -> FactorReader {
        unexpanded!()
    }

    /// [`mul`](Tile::mul) by an optional level; absent, it scales by one.
    pub fn mul_bound<S: Numeric>(&self, _level: &ComptimeOption<Tile<S>>) -> Tile<E> {
        unexpanded!()
    }

    /// These values read placed: an `e2m1` field the device emulates decoded without its lift,
    /// which the factor this tile is read under carries instead ([`reader`](Tile::reader)), one
    /// multiply a reader rather than one a value. What a leaf that applies the factor to every
    /// value it reads takes; any other tile as it is.
    pub(crate) fn placed(&self) -> Tile<E> {
        unexpanded!()
    }

    /// These values as indices into `table`, decoded only by [`copy_from`](Tile::copy_from).
    pub fn lookup<S: Numeric>(&self, _table: &Tile<S>) -> Tile<E> {
        unexpanded!()
    }

    pub(crate) fn refuse_factor(&self, _site: &str) {
        unexpanded!()
    }

    /// Whether these values carry scales ([`mul`](Tile::mul)).
    pub(crate) fn scaled(&self) -> comptime_type!(bool) {
        unexpanded!()
    }
}

impl<E: Numeric> TileExpand<E> {
    pub fn __expand_mul_method<S: Numeric>(
        &self,
        scope: &Scope,
        scale: &TileExpand<S>,
    ) -> TileExpand<E> {
        let values = self.place.space.clone();
        let scales = scale.place.space.clone();
        assert!(
            scales.axes().all(|axis| values.contains(axis)),
            "Tile::mul: the scales span {:?} where the values span {:?}; a scale is looked up at \
             the value's coordinates, so every axis of the scales is one of the values'",
            scales.axes().collect::<Vec<_>>(),
            values.axes().collect::<Vec<_>>()
        );
        let mut out = self.clone();
        match &mut out.kind {
            TileKindExpand::Memory(memory) => {
                memory.factor = memory.factor.and(FactorExpand::of(scope, scale));
            }
            TileKindExpand::PlaneTile(_)
            | TileKindExpand::PlanePartition(_)
            | TileKindExpand::TmaGmem(_)
            | TileKindExpand::Procedural(_)
            | TileKindExpand::Lines(_) => {
                panic!(
                    "Tile::mul: a factor rides values read from memory; a fragment is scaled \
                     once it holds them and a tma source is not read here at all"
                )
            }
        }
        out
    }

    pub(crate) fn __expand_reader_method(
        &self,
        scope: &Scope,
        axes: MatrixAxes,
        matrix: NativeExpand<usize>,
        side: Side,
        out: Space,
        acc_axes: MatrixAxes,
    ) -> FactorReaderExpand {
        refuse_codebook(self, "a leaf's read");
        let values = self.place.space.clone();
        // A line is the run of a load along the innermost axis; a load stored across several
        // columns is read as their runs, each placed at its own column.
        let vector_size = self.clone().__expand_vector_tile_method(scope).extents()[0].1;
        let factor = match &self.kind {
            TileKindExpand::Memory(memory) => memory.factor.clone(),
            _ => FactorExpand::default(),
        };
        if let Some(inner) = factor.inner() {
            check_scales_omit_rather_than_divide(&inner.projection);
            check_scales_ride(side, &inner.space, &out, acc_axes);
            let innermost = values.axis_at(values.rank() - 1);
            assert!(
                vector_size == 1 || !inner.projection.addresses(innermost),
                "Tile::mul: a line runs {vector_size} values along {innermost:?}, which its \
                 scales address, so one line lies under several scales; serve the factor one \
                 value a line, or omit {innermost:?} from the scales"
            );
        }
        // A tile read placed carries its values' lift in its factor: one multiply of the coarse
        // levels, met once, and every value read under the factor even where it has no scale.
        let lift = self.__expand_packing_method(scope).lift();
        let coarse = match lift == 1.0 {
            true => factor.coarse(scope),
            false => {
                let lift: NativeExpand<f32> =
                    cubecl::ir::ExpandValue::constant((lift as u64).into(), f32::elem_type(scope))
                        .into();
                MulExpand::__expand_mul_method(factor.coarse(scope), scope, lift)
            }
        };
        FactorReaderExpand {
            coarse,
            inner: factor.innermost(),
            scaled: factor.scaled() || lift != 1.0,
            values,
            axes,
            vector_size,
            matrix,
        }
    }

    pub fn __expand_mul_bound_method<S: Numeric>(
        &self,
        scope: &Scope,
        level: &ComptimeOptionExpand<Tile<S>>,
    ) -> TileExpand<E> {
        match level {
            ComptimeOptionExpand::Some(level) => self.__expand_mul_method(scope, level),
            ComptimeOptionExpand::None => self.clone(),
        }
    }

    pub(crate) fn __expand_placed_method(&self, _scope: &Scope) -> TileExpand<E> {
        let mut out = self.clone();
        if let TileKindExpand::Memory(memory) = &mut out.kind {
            memory.store.packing = memory.store.packing.placed();
        }
        out
    }

    pub(crate) fn __expand_scaled_method(&self, _scope: &Scope) -> bool {
        match &self.kind {
            TileKindExpand::Memory(memory) => memory.factor.scaled() || memory.codebook.present(),
            _ => false,
        }
    }

    pub fn __expand_lookup_method<S: Numeric>(
        &self,
        _scope: &Scope,
        table: &TileExpand<S>,
    ) -> TileExpand<E> {
        assert!(
            table.place.space.rank() == 1,
            "Tile::lookup: a table is a tile of one axis, its positions; this one spans {:?}",
            table.place.space.axes().collect::<Vec<_>>()
        );
        let mut out = self.clone();
        match &mut out.kind {
            TileKindExpand::Memory(memory) => {
                let packing = memory.store.packing;
                let Packing::Packed {
                    field: Field::Index { bits },
                } = packing
                else {
                    panic!(
                        "Tile::lookup: the values index a table through an index field \
                         (`Field::Index`), and these are stored as {packing:?}"
                    )
                };
                assert!(
                    table.place.space.extent_at(0) >= 1 << bits,
                    "Tile::lookup: a {bits}-bit index names {} positions; the table holds {}",
                    1usize << bits,
                    table.place.space.extent_at(0)
                );
                assert!(
                    !memory.codebook.present(),
                    "Tile::lookup: these values already index a table"
                );
                memory.codebook = CodebookExpand::of(table);
            }
            TileKindExpand::PlaneTile(_)
            | TileKindExpand::PlanePartition(_)
            | TileKindExpand::TmaGmem(_)
            | TileKindExpand::Procedural(_)
            | TileKindExpand::Lines(_) => {
                panic!("Tile::lookup: a table is indexed by values stored in memory")
            }
        }
        out
    }

    pub(crate) fn __expand_refuse_factor_method(&self, _scope: &Scope, site: &str) {
        refuse_codebook(self, site);
        if let TileKindExpand::Memory(memory) = &self.kind {
            assert!(
                !memory.factor.scaled(),
                "{site}: this leaf takes its operands from registers, where scales have nowhere                  to land; contract through a fragment or in memory"
            );
        }
    }
}

#[cube]
impl<S: Numeric> Tile<S> {
    /// Whether a read of this tile is a plane shuffle, so readers must keep units converged.
    #[allow(dead_code)] // Reached through its expand, from [`FactorRead`].
    pub(crate) fn by_shuffle(&self) -> comptime_type!(bool) {
        match &self.kind {
            TileKind::Lines(_) => comptime!(true),
            TileKind::Memory(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => comptime!(false),
        }
    }

    /// The one scale at `coords`, one entry per axis of this tile's space.
    #[allow(dead_code)] // Reached through its expand, from [`FactorRead`].
    pub(crate) fn scale_at(&self, coords: &Coords<u32>) -> S {
        match &self.kind {
            TileKind::Lines(lines) => lines.read(coords),
            TileKind::Memory(_) => self.value_in_line(coords),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                panic!("Tile::scale_at: a scale is read from memory or from the plane's units")
            }
        }
    }

    /// The one scale covering the value at `coords` of a tile spanning `values`.
    #[allow(dead_code)] // Reached through its expand, from [`FactorRead`].
    pub(crate) fn scale_for(&self, coords: &Coords<u32>, #[comptime] values: Space) -> S {
        let rank = comptime!(self.place.space.rank());
        let mut own = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            let axis = comptime!(self.place.space.axis_at(p));
            own.push(coords.at(comptime!(values.position(axis))));
        }
        self.scale_at(&own)
    }

    /// The one value at `coords` of a tile that serves lines: the load holding it, read whole,
    /// and the value's place in it, whatever axes the load spans.
    #[allow(dead_code)] // Reached through its expand, from [`Tile::scale_at`].
    fn value_in_line(&self, coords: &Coords<u32>) -> S {
        let space = comptime!(self.place.space.clone());
        let rank = comptime!(space.rank());
        let load = self.vector_tile();
        let width = comptime!(load.values());
        let size!(W) = width;
        let line = self
            .nd_packed::<W>(comptime!(Guard::Checked))
            .read(load.index(coords, &space));
        if comptime!(width > 1) {
            let mut field = 0u32.runtime();
            #[unroll]
            for p in 0..rank {
                let axis = comptime!(space.axis_at(p));
                let extent = comptime!(load.extent_along(axis) as u32);
                if comptime!(extent > 1) {
                    let step = comptime!(load.step_along(axis) as u32);
                    field += coords.at(p).remainder(extent).times(step);
                }
            }
            line.extract_dynamic(field.cast::<usize>())
        } else {
            line.extract(0usize)
        }
    }
}

#[cube]
impl<E: Float> Tile<E> {
    /// This tile's cells in order, one value a slice of something else, as the rows of a softmax:
    /// a window of memory a plane holds, `count` cells, every unit of the plane reading them all.
    pub(crate) fn cells(&self, #[comptime] count: usize) -> Array<E> {
        comptime!(assert!(
            self.place.holder() == ComputeScope::Plane && self.place.space.cells() == count,
            "Tile::cells: a plane's window of {count} cells, not {:?} held by {:?}",
            self.place.space,
            self.place.holder()
        ));
        match &self.kind {
            TileKind::Memory(window) => window.cells(count),
            TileKind::PlanePartition(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => Tile::<E>::refuse_cells(),
        }
    }

    /// Write `values` over this tile's `count` cells ([`cells`](Tile::cells)), from the plane's
    /// first unit.
    pub(crate) fn set_cells(&mut self, values: &Array<E>, #[comptime] count: usize) {
        comptime!(assert!(
            self.place.holder() == ComputeScope::Plane && self.place.space.cells() == count,
            "Tile::cells: a plane's window of {count} cells, not {:?} held by {:?}",
            self.place.space,
            self.place.holder()
        ));
        match &mut self.kind {
            TileKind::Memory(window) => window.set_cells(values, count),
            TileKind::PlanePartition(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_)
            | TileKind::Lines(_) => Tile::<E>::refuse_cells(),
        }
    }

    fn refuse_cells() -> ! {
        panic!("Tile::cells: a plane's window of memory holds the cells every unit of it reads")
    }
}

/// Refuses values that index a table ([`Tile::lookup`]) at `site`.
pub(crate) fn refuse_codebook<E: Numeric>(tile: &TileExpand<E>, site: &str) {
    if let TileKindExpand::Memory(memory) = &tile.kind {
        assert!(
            !memory.codebook.present(),
            "{site}: these values are indices into a table (`Tile::lookup`), which only a copy \
             decodes; copy them into a stage first (`stage.copy_from(&w.lookup(&table))`)"
        );
    }
}
