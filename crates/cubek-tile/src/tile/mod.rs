//! The [`Tile`]: one operand's data as a [`TileKind`] backing store, plus the comptime
//! [`Space`] it projects. Structure only; each store's own data and leaves live in its file
//! ([`mem`], [`cmma`], [`tma`]). The launch surface (specs, deliveries, builder) lives in
//! `physical/`; a kernel's first line is [`Tile::of`] on a plain tensor.

mod cmma;
mod mem;
mod mma;
mod plane;
mod procedural;
mod register;
mod tma;
mod view;

pub use cmma::*;
pub use mem::*;
pub use mma::*;
pub use plane::*;
pub use procedural::*;
pub use register::*;
pub use tma::*;
pub use view::*;

use cubecl::{ir::Scope, unexpanded};
use cubecl::{
    prelude::*,
    quant::scheme::QuantScheme,
    std::quant::view::{KnownScale, QuantizedView as DequantView},
    std::tensor::{View, layout::Coordinates},
};

use crate::*;

impl<T: Numeric> Tile<T> {
    /// Create a coordinate-backed tile from an arbitrary procedural recipe. The concrete recipe
    /// is erased only while CubeCL expands this call. `leaf` is what will consume the tile, which
    /// a recipe has no spec to carry: it is the leaf the peer operands state.
    pub fn procedural<R: Recipe<T> + 'static>(_space: Space, _recipe: R, _leaf: Leaf) -> Self {
        unexpanded!()
    }

    pub fn __expand_procedural<R: Recipe<T> + 'static>(
        scope: &Scope,
        space: Space,
        recipe: R::ExpandType,
        leaf: Leaf,
    ) -> TileExpand<T> {
        Self::__expand_procedural_resident::<R>(scope, space, recipe, StagePlan::in_place(), leaf)
    }

    /// [`procedural`](Tile::procedural) with the residences stated: a level asking for a stage
    /// cooperatively materializes the recipe into it, which is how a source with no bytes reaches
    /// a leaf that cannot evaluate one.
    pub fn procedural_resident<R: Recipe<T> + 'static>(
        _space: Space,
        _recipe: R,
        _stage: StagePlan,
        _leaf: Leaf,
    ) -> Self {
        unexpanded!()
    }

    pub fn __expand_procedural_resident<R: Recipe<T> + 'static>(
        scope: &Scope,
        space: Space,
        recipe: R::ExpandType,
        stage: StagePlan,
        leaf: Leaf,
    ) -> TileExpand<T> {
        Self::__expand_procedural_virtual(
            scope,
            space,
            VirtualRecipe::<T>::__expand_new::<R>(scope, recipe),
            stage,
            leaf,
        )
    }

    /// Create a coordinate-backed tile yielding constant zero.
    pub fn zeros(_space: Space, _leaf: Leaf) -> Self {
        unexpanded!()
    }

    pub fn __expand_zeros(scope: &Scope, space: Space, leaf: Leaf) -> TileExpand<T> {
        Self::__expand_procedural::<Zeros>(scope, space, ZerosExpand {}, leaf)
    }

    /// Create a coordinate-backed tile yielding constant one.
    pub fn ones(_space: Space, _leaf: Leaf) -> Self {
        unexpanded!()
    }

    pub fn __expand_ones(scope: &Scope, space: Space, leaf: Leaf) -> TileExpand<T> {
        Self::__expand_procedural::<Ones>(scope, space, OnesExpand {}, leaf)
    }

    /// Create a coordinate-backed tile yielding a constant value.
    pub fn constant(_space: Space, _value: T, _leaf: Leaf) -> Self {
        unexpanded!()
    }

    pub fn __expand_constant(
        scope: &Scope,
        space: Space,
        value: NativeExpand<T>,
        leaf: Leaf,
    ) -> TileExpand<T> {
        Self::__expand_procedural::<Constant<T>>(scope, space, ConstantExpand::<T> { value }, leaf)
    }
}

/// A tile's backing store. Every variant is lifetime-free (a `Box<[T]>` or a
/// [`cmma::Matrix`](cubecl::cmma::Matrix)); [`view`](Tile::view) rebuilds a borrowed view on
/// demand.
///
/// `Clone` copies the handle, not the cells: two clones name the same storage, so writing through
/// one is visible through the other. That is how later ring slots reuse a fixed operand's first
/// buffer, and it is only sound where nothing rewrites the buffer afterwards.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum TileKind<T: Numeric> {
    Gmem(MemData<T>),
    Smem(MemData<T>),
    /// One plane-level tile: owned by a plane and sliced across its lanes, so never addressable
    /// (no memory view). The [`Leaf`] picks the encoding; the contraction is its own.
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

/// Where an operand's quantized form is decoded: the one site that turns stored values into served
/// ones. Stated at launch, once, since the quantized form ends at exactly one boundary. Which sites
/// are available is fixed by what the operand's transports can decode, never by preference, so a
/// stated value is either the one that was left (which
/// [`build`](crate::StridedTileSource::build) enforces) or a genuine fork between stage size and
/// per-read cost.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum DequantAt {
    /// The load into the stage decodes; the stage holds served values, so it inflates by the
    /// served-to-stored ratio and the achievable stage depth drops with it.
    Load,
    /// The stage keeps the quantized values and their scales; the instruction's read decodes,
    /// amortized over whatever reuse the leaf has.
    Read,
}

/// Quantization a tile's store carries, so reads dequantize on their own. Holds the scale `buffer`
/// plus what walks the scales in step with the values: a per-axis `strides`, a running
/// `window_start`, and comptime `block` sizes. [`ScaleLayout`] turns those into an address ([`MemData::at`]).
/// Per-tensor is the trivial case: one scale, every stride `0`, `window_start` never moves.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct QuantInfo {
    pub(crate) buffer: Box<[f32]>,
    /// What every read below this window already holds of its scale, settled once at
    /// construction: the global level's scale read from its binding, or nothing. Never
    /// [`KnownScale::Whole`] here: a stage's scales do not exist until its fill, so a uniform
    /// window promotes to it at read time ([`dequant_view`](QuantInfo::dequant_view)).
    pub(crate) known: KnownScale,
    pub(crate) strides: Coords<u32>,
    pub(crate) window_start: u32,
    #[cube(comptime)]
    pub(crate) block: Vec<usize>,
    /// Per-axis extent of the window these scales cover, in elements; [`usize::MAX`] where it is
    /// not comptime (a dynamic top-level axis). An axis whose extent fits inside a block has no
    /// distinct scales left to address, which is what [`ScaleLayout`] drops its term for.
    #[cube(comptime)]
    pub(crate) extent: Vec<usize>,
    /// Where this operand's quantized form ends. Read by [`MemData::smem_like`], which is why no
    /// call site asks an operand whether it is quantized before staging it.
    #[cube(comptime)]
    pub(crate) dequant_at: DequantAt,
    /// Per-axis count of distinct scales the buffer holds, set only on a *staged* smem side-channel
    /// ([`MemData::smem_quant`]): the values stage as packed words and their scales stage compactly
    /// beside them, so the fill knows how many blocks to copy. Empty for a gmem operand, which reads
    /// the tensor's own scales in place.
    #[cube(comptime)]
    pub(crate) scale_shape: Vec<usize>,
    /// A lookup scheme's `2^bits`-entry table, present exactly under
    /// [`QuantMode::Lookup`](cubecl::quant::scheme::QuantMode). Always the gmem buffer: it is at
    /// most a few hundred cache-resident floats, so a stage carries it through rather than
    /// copying it ([`smem_quant`](MemData::smem_quant)).
    pub(crate) table: ComptimeOption<Box<[f32]>>,
    #[cube(comptime)]
    pub scheme: QuantScheme,
}

/// Per-axis block edges (elements per block) for a scheme. Per-tensor is one scale for the whole
/// tensor, so every axis reports `usize::MAX`: with `0` strides ([`Tile::of_dequant`]) the value
/// never addresses a real block, and it makes [`uniform_window`] report the whole window as
/// uniform, which per-tensor always is. A block scheme's edges come straight from the scheme.
pub(crate) fn block_edges(scheme: QuantScheme, rank: usize) -> Vec<usize> {
    let Some(block) = scheme.block_size() else {
        return vec![usize::MAX; rank];
    };
    block.to_dim_vec(rank).iter().map(|&b| b as usize).collect()
}

/// Whether one scale covers a window of `extent` under `block` edges: every axis fits inside a
/// block, so there is nothing left for [`ScaleLayout`] to address and the scale can be read once
/// ([`QuantInfo::uniform_scale`]) instead of per value.
fn uniform_window(block: &[usize], extent: &[usize]) -> bool {
    (0..block.len()).all(|p| extent[p] <= block[p])
}

impl QuantInfo {
    /// See [`uniform_window`]. Both this and its expand twin exist because a `comptime!` branch
    /// typechecks as host code as well as expanded.
    pub(crate) fn uniform(&self) -> bool {
        uniform_window(&self.block, &self.extent)
    }
}

impl QuantInfoExpand {
    /// See [`uniform_window`].
    pub(crate) fn uniform(&self) -> bool {
        uniform_window(&self.block, &self.extent)
    }
}

/// Per-axis window extent in elements for a space's own level, [`usize::MAX`] where an axis is
/// dynamic. What [`QuantInfo`] carries so [`ScaleLayout`] can drop the axes that hold one scale.
pub(crate) fn window_extents(space: &Space, rank: usize) -> Vec<usize> {
    (0..rank)
        .map(|p| match space.extent_raw(space.axis_at(p)) {
            Extent::Static(e) => e,
            Extent::Dynamic => usize::MAX,
        })
        .collect()
}

/// The scheme a staged side-channel serves: its grid holds *effective* scales
/// ([`MemData::stage_scales`] folds the global level in), so a two-level scheme stages as its
/// one-level block form and reads below the stage carry no global scale.
pub(crate) fn staged_scheme(scheme: QuantScheme) -> QuantScheme {
    let Some(block) = scheme.block_scale() else {
        return scheme;
    };
    // Rebuilt rather than cleared: the levels are set additively and there is no way to drop one.
    QuantScheme::default()
        .with_value(scheme.value)
        .with_store(scheme.store)
        .with_mode(scheme.mode)
        .per_block(block.size.as_slice(), block.dtype)
}

#[cube]
impl QuantInfo {
    /// The one scale this whole window reconstructs against, global level folded in. Only
    /// meaningful where [`uniform`](QuantInfoExpand::uniform) holds; one load for the whole tile.
    pub(crate) fn uniform_scale(&self) -> f32 {
        self.known
            .effective(self.buffer[self.window_start.fcast::<usize>()])
    }

    /// The [`DequantView`] this info's scale data resolves to for a values/scales view pair over
    /// the same coordinates: a uniform window promotes to one whole scale, read here so no read
    /// below pays for the scales view at all; any other window reads with what it already
    /// [`known`](QuantInfo::known). Shared by [`flat_transparent`](MemData::flat_transparent) and
    /// [`transparent`](MemData::transparent).
    pub(crate) fn dequant_view<
        'a,
        I: Numeric,
        WP: Size,
        T: Numeric,
        W: Size,
        C: Coordinates + 'static,
    >(
        &self,
        values: View<'a, Vector<I, WP>, C>,
        scales: View<'a, f32, C>,
    ) -> DequantView<'a, I, WP, f32, T, W, C> {
        let known = if comptime!(self.uniform()) {
            KnownScale::new_Whole(self.uniform_scale())
        } else {
            self.known
        };
        DequantView::<I, WP, f32, T, W, C>::new_with_known_scale(
            values,
            scales,
            known,
            self.table.clone(),
            comptime!(self.scheme),
        )
    }

    /// Re-window the scales onto a tile whose absolute logical origin is `origin`. Per axis the block
    /// index is `origin / block`, dotted with the scale strides and summed into a flat start (elements
    /// everywhere, the inner axis scaled back by `vector_size`; per-tensor keeps strides `0`). Folding
    /// the window's own block index in here lets [`ScaleLayout`] add only the within-window offset,
    /// sound because a window never straddles a block (`validate_scheme` enforces it).
    pub(crate) fn window(
        &self,
        origin: &Coords<u32>,
        #[comptime] rank: usize,
        #[comptime] vector_size: usize,
        #[comptime] extent: Vec<usize>,
    ) -> QuantInfo {
        let last = comptime!(rank - 1);
        let mut advances = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            let w = comptime!(if p == last { vector_size } else { 1usize });
            let origin_elem = origin.at(p).fmul(comptime!(w as u32).runtime());
            let block = comptime!(self.block[p] as u32).runtime();
            advances.push(origin_elem.fdiv(block).fmul(self.strides.at(p)));
        }
        QuantInfo {
            buffer: unsafe { self.buffer.as_boxed_unchecked() },
            known: self.known,
            strides: self.strides.clone(),
            window_start: advances.fsum(comptime!((0..rank).collect::<Vec<_>>())),
            block: comptime!(self.block.clone()),
            extent: comptime!(extent),
            dequant_at: comptime!(self.dequant_at),
            scale_shape: comptime!(self.scale_shape.clone()),
            table: self.table.clone(),
            scheme: comptime!(self.scheme),
        }
    }
}

/// One operand's data: a runtime [`TileKind`] backing store, the comptime [`Space`] it projects,
/// and what it is at the instruction ([`Leaf`]). `T` is the element the tile serves and computes in;
/// its physical vector width is a storage detail inside the [`TileKind`], read back with
/// [`vector_size`](Tile::vector_size).
///
/// The leaf rides here, not on the [`Space`]: it is a format decision, and formats belong to the
/// operand whose format they are. The partitioning says how the problem is cut and nothing about
/// what the pieces become. Operands that disagree meet the kind-pairing panics at the instruction,
/// which is the same way every other mismatched pair is caught.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Tile<T: Numeric> {
    pub tile_kind: TileKind<T>,
    #[cube(comptime)]
    pub space: Space,
    #[cube(comptime)]
    pub leaf: Leaf,
}

/// The one physical dim whose bound is `axis`'s own extent: it carries `axis` alone, at
/// coefficient `1`. `None` otherwise, which is either of the two ways a bound stops being that
/// extent: a gather, where the dim holds the receptive field several axes reach over, and storage
/// tiling, where the extent is the product over the dims the axis is split across.
fn bound_states(projection: &Projection, axis: Axis) -> Option<usize> {
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
    /// Whether the leaf can consume this operand in its current physical form. Opaque fragment
    /// transports require shared memory or an already materialized fragment; scalar and manual
    /// readers can address their source directly.
    fn reads_in_place(&self) -> comptime_type!(bool) {
        match &self.tile_kind {
            TileKind::Smem(_) | TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                comptime!(true)
            }
            TileKind::Gmem(_) | TileKind::TmaGmem(_) | TileKind::Procedural(_) => {
                comptime!(match self.leaf {
                    Leaf::Memory { .. } => true,
                    Leaf::Mma { io } => {
                        matches!(io.lhs_load_method, LoadMethod::Manual)
                            && matches!(io.rhs_load_method, LoadMethod::Manual)
                    }
                    Leaf::Cmma => false,
                })
            }
        }
    }

    /// Create a scalar, memory-free tile over a logical space, evaluated where it is read at every
    /// level. Dynamic extents are supplied by another operand when an operation is walked; a
    /// procedural tile never witnesses them. `leaf` is what will consume it, stated here like
    /// every other operand states it.
    fn procedural_virtual(
        #[comptime] space: Space,
        recipe: VirtualRecipe<T>,
        #[comptime] stage: StagePlan,
        #[comptime] leaf: Leaf,
    ) -> Self {
        Tile::<T> {
            tile_kind: TileKind::new_Procedural(ProceduralData::<T>::new_virtual(
                comptime!(space.clone()),
                recipe,
                stage,
            )),
            space,
            leaf,
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
    pub fn stage_source(&self) -> comptime_type!(StageSource) {
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
                    "Tile::residence: a {:?} leaf cannot read this operand's current physical \
                     form in place; materialize it with Residence::Smem at \
                     some level above the leaf",
                    self.leaf
                );
            });
            comptime!(Residence::InPlace)
        } else {
            let procedural = self.is_procedural();
            comptime!(
                if procedural && matches!(requested, Residence::Register(_)) {
                    panic!(
                        "Tile::residence: a procedural source has no plane-fragment transport; state \
                     Residence::Smem to materialize it into shared memory, or Residence::InPlace \
                     to evaluate it at the leaf"
                    );
                }
            );
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
    /// from — the split is the space's, not the storage's.
    pub(crate) fn lane_share(&self) -> comptime_type!(LaneShare) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.lane_share,
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(LaneShare::Whole)
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

    /// Comptime quant dispatch for a leaf read (`0` = plain, `1` = native i8, `>1` = packed u32);
    /// see [`MemData::quant_pack`]. A resident fragment and a tma source are never quantized.
    pub(crate) fn quant_pack(&self) -> comptime_type!(usize) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.quant_pack(),
            TileKind::TmaGmem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::Procedural(_) => {
                comptime!(0usize)
            }
        }
    }

    /// Whether this tile's buffer is addressed through a non-identity [`Projection`]: a logical
    /// coordinate is not a physical one, so the only read surface that describes the tile is the
    /// N-D one ([`nd`](Tile::nd)) and no window of it is dense. False for a
    /// [`direct`](Projection::direct) operand, and for a fragment or a tensor map, which have no
    /// buffer to gather from.
    pub fn gathered(&self) -> comptime_type!(bool) {
        let projection = self.projection();
        comptime!(!projection.is_direct())
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

    /// Whether this tile can state `axis`'s runtime size for the operation it takes part in: it
    /// spans the axis [`Dynamic`](crate::Extent), has a buffer to read a bound off, and that bound
    /// is the axis's own extent ([`bound_states`]). Each clause rules out an operand that spans the
    /// axis without being able to answer for it: a `Static` one knows its size at comptime already
    /// (and a broadcast `1` is not the operation's extent), a fragment has no bound, and a gathered
    /// axis's bound is the receptive field its axes reach over.
    ///
    /// Spanning an axis and having to supply it are therefore separate questions: an operation
    /// sizes each `Dynamic` axis from whichever of its operands witnesses it and lets the others
    /// pass ([`witnessed_space`]).
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
            leaf: comptime!(self.leaf),
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

    /// This operand's runtime logical size along `axis`, read off the [`bound`](MemData)
    /// folded from the tensor shape. The source of a [`Dynamic`](crate::Extent) axis's
    /// tile count. Only an axis this tile [`witnesses`](Tile::witnesses) has one: a fragment has no
    /// buffer extent ([`bounded`](Tile::bounded)), and a gathered or storage-tiled axis no bound of
    /// its own ([`bound_position`]).
    pub fn runtime_extent(&self, #[comptime] axis: Axis) -> usize {
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
    /// own width): index `i` reads line `origin + i` — one add, no layout
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
    pub fn dense_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
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
                    s.store_window(d)
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

    /// Drain a resident accumulator into memory `dst`, casting `T` down to `dst`'s element
    /// type. [`copy_from`](Self::copy_from) can't: its transports move bytes so stay same-type,
    /// but a register accumulator (`f32`) is wider than the output it writes (`f16`). Only a
    /// fragment partition drains this way.
    pub fn drain_cast_into<Out: Numeric>(&self, dst: &mut Tile<Out>) {
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
