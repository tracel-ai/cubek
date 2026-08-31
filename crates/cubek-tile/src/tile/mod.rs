//! The [`Tile`]: one operand's data as a [`TileKind`] backing store, plus the comptime
//! [`Space`] it projects. Structure only; each store's own data and leaves live in its file
//! ([`mem`], [`cmma`], [`tma`]). The launch surface (specs, deliveries, builder) lives in
//! `physical/`; a kernel's first line is [`Tile::of`] on a plain tensor.

mod cmma;
mod geometry;
mod mem;
mod mma;
mod packing;
mod plane;
mod procedural;
mod register;
mod tma;
mod view;

pub use cmma::*;
pub use geometry::*;
pub use mem::*;
pub use mma::*;
pub use packing::*;
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
    std::tensor::{
        View,
        layout::{Coordinates, CoordsDyn},
    },
};

use cubecl::zspace::SmallVec;

use crate::*;

impl<T: Numeric> Tile<T> {
    /// Create a coordinate-backed tile from an arbitrary procedural recipe. The concrete recipe
    /// is erased only while CubeCL expands this call.
    pub fn procedural<R: Recipe<T> + 'static>(_space: Space, _recipe: R) -> Self {
        unexpanded!()
    }

    pub fn __expand_procedural<R: Recipe<T> + 'static>(
        scope: &Scope,
        space: Space,
        recipe: R::ExpandType,
    ) -> TileExpand<T> {
        Self::__expand_procedural_resident::<R>(scope, space, recipe, StagePlan::in_place())
    }

    /// Create a procedural tile while preserving the recipe's factorization for contraction: the
    /// consumer sees one factor per contracted axis instead of one opaque field.
    pub fn procedural_separable<R: SeparableRecipe<T> + 'static>(_space: Space, _recipe: R) -> Self
    where
        R::ExpandType: SeparableRecipeAxisDependencies,
    {
        unexpanded!()
    }

    pub fn __expand_procedural_separable<R: SeparableRecipe<T> + 'static>(
        scope: &Scope,
        space: Space,
        recipe: R::ExpandType,
    ) -> TileExpand<T>
    where
        R::ExpandType: SeparableRecipeOps<T>,
    {
        // A separable procedural tile is always evaluated in place. This invariant is load-bearing
        // for normalization: staging a recipe into shared memory would drop its factorization and
        // normalization metadata without diagnostic.
        Self::__expand_procedural_virtual(
            scope,
            space,
            VirtualRecipe::<T>::__expand_new_separable::<R>(scope, recipe),
            StagePlan::in_place(),
        )
    }

    /// [`procedural`](Tile::procedural) with the residences stated: a level asking for a stage
    /// cooperatively materializes the recipe into it, which is how a source with no bytes reaches
    /// a reader that cannot evaluate one.
    pub fn procedural_resident<R: Recipe<T> + 'static>(
        _space: Space,
        _recipe: R,
        _stage: StagePlan,
    ) -> Self {
        unexpanded!()
    }

    pub fn __expand_procedural_resident<R: Recipe<T> + 'static>(
        scope: &Scope,
        space: Space,
        recipe: R::ExpandType,
        stage: StagePlan,
    ) -> TileExpand<T> {
        Self::__expand_procedural_virtual(
            scope,
            space,
            VirtualRecipe::<T>::__expand_new::<R>(scope, recipe),
            stage,
        )
    }

    /// Create a coordinate-backed tile yielding constant zero.
    pub fn zeros(_space: Space) -> Self {
        unexpanded!()
    }

    pub fn __expand_zeros(scope: &Scope, space: Space) -> TileExpand<T> {
        Self::__expand_procedural::<Zeros>(scope, space, ZerosExpand {})
    }

    /// Create a coordinate-backed tile yielding constant one.
    pub fn ones(_space: Space) -> Self {
        unexpanded!()
    }

    pub fn __expand_ones(scope: &Scope, space: Space) -> TileExpand<T> {
        Self::__expand_procedural::<Ones>(scope, space, OnesExpand {})
    }

    /// Create a coordinate-backed tile yielding a constant value.
    pub fn constant(_space: Space, _value: T) -> Self {
        unexpanded!()
    }

    pub fn __expand_constant(scope: &Scope, space: Space, value: NativeExpand<T>) -> TileExpand<T> {
        Self::__expand_procedural::<Constant<T>>(scope, space, ConstantExpand::<T> { value })
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

/// The [`Packing`] a quantization scheme implies: how many of its values a stored element holds
/// and what field each occupies. The one place a scheme is read for a fact about *storage*, so a
/// quantized operand and one that merely states [`TileSpec::packed`] answer every reader alike.
pub(crate) fn scheme_packing(scheme: QuantScheme) -> Packing {
    match scheme.num_quants() {
        1 => Packing::Native,
        _ => Packing::Packed {
            field: scheme.value,
        },
    }
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

/// What an out-of-range table entry means.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IndexPolicy {
    /// The launch guarantees every entry addresses a live slice of the target axis; no test is
    /// emitted and the window is placed wherever the entry says.
    Trusted,
    /// The displaced window is masked against the target axis's own bound, so an entry past it
    /// reads as the window's [`Boundary`]. The launch-side builder arms that axis directly from
    /// this policy; a hand-built [`TileSpec`] must carry the corresponding boundary itself.
    Checked,
}

/// How an [`Indirection`] interprets its table entries.
///
/// Replacement routing selects independently placed slices or pages. A sequence range instead
/// translates one logical token axis into the packed interval delimited by two adjacent cumulative
/// lengths. Keeping the policy inside `Replace` makes a sequence range intrinsically checked: its
/// end entry is the logical boundary, not an optional safety policy.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum IndirectionMode {
    /// `table[..., t / granularity]` names the physical entry holding logical target coordinate
    /// `t`, and the lookup replaces the window's origin with it.
    Replace {
        /// Elements of `target` per table entry: a page size for a paged cache, `1` for a slice
        /// index. Comptime so the divide folds and the divisibility rules are checkable on the
        /// host.
        granularity: usize,
        /// What an entry past the target axis's bound means. Only replacement routing has the
        /// choice; see [`IndexPolicy`].
        policy: IndexPolicy,
    },
    /// Adjacent entries `table[b]` and `table[b + 1]` are the physical start and exclusive end of
    /// the sequence at coordinate `b` of the single index axis. The lookup translates the target
    /// window to that start and narrows its bound to that end.
    SequenceRange,
}

/// The comptime half of an [`Indirection`]: everything that decides *where* to look, with no
/// buffer. Part of a kernel's identity, so different routing semantics never share a compiled
/// kernel.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct IndirectionSpec {
    /// The operand axis whose window origin the lookup displaces.
    pub target: Axis,
    /// Which axes' region coordinates address the table, outer to inner. A sequence range takes
    /// exactly one, the axis its cumulative lengths are indexed by.
    pub index_axes: SmallVec<[Axis; MAX_AXES]>,
    /// What a table entry means, and so what the lookup does when it fires.
    pub mode: IndirectionMode,
}

impl IndirectionSpec {
    /// A per-tile slice index: one entry per `index_axis` tile, displacing `target`'s origin to
    /// the entry outright (`granularity = 1`). MoE expert routing is exactly this.
    pub fn indexed(index_axis: Axis, target: Axis, policy: IndexPolicy) -> Self {
        IndirectionSpec {
            target,
            index_axes: SmallVec::from_slice(&[index_axis]),
            mode: IndirectionMode::Replace {
                granularity: 1,
                policy,
            },
        }
    }

    /// A paged slice: `page_size` consecutive elements of `target` share one entry, so the table
    /// places whole pages and a position inside a page carries through untouched. A paged KV
    /// cache is this, with `index_axis` the batch, `target` the absolute token position, and the
    /// table its page table.
    ///
    /// `page_size` is comptime, so the entry divide folds; a power of two folds to a shift. It
    /// also bounds where the lookup may fire: [`validate`](Self::validate) refuses a level that
    /// cuts `target` into windows that are not whole pages, and one below the fire level that
    /// would start mid-page.
    pub fn paged(index_axis: Axis, target: Axis, page_size: usize, policy: IndexPolicy) -> Self {
        assert!(
            page_size > 0,
            "IndirectionSpec::paged: a page holds at least one element"
        );
        IndirectionSpec {
            target,
            index_axes: SmallVec::from_slice(&[index_axis]),
            mode: IndirectionMode::Replace {
                granularity: page_size,
                policy,
            },
        }
    }

    /// A variable-length packed sequence. Adjacent entries in `cumulative_lengths` delimit the
    /// physical target-axis range for one coordinate of `sequence_axis`: entry `b` is its start
    /// and entry `b + 1` its exclusive end. The range translates the target origin and always
    /// masks at its end.
    pub fn variable_length(sequence_axis: Axis, target: Axis) -> Self {
        IndirectionSpec {
            target,
            index_axes: SmallVec::from_slice(&[sequence_axis]),
            mode: IndirectionMode::SequenceRange,
        }
    }

    /// The policy attached to replacement routing. A sequence range has no policy: its exclusive
    /// end is necessarily a checked boundary.
    pub(crate) fn replacement_policy(&self) -> Option<IndexPolicy> {
        match self.mode {
            IndirectionMode::Replace { policy, .. } => Some(policy),
            IndirectionMode::SequenceRange => None,
        }
    }

    /// Whether this lookup translates a sequence range rather than replacing a slice or page.
    pub(crate) fn is_sequence_range(&self) -> bool {
        matches!(self.mode, IndirectionMode::SequenceRange)
    }

    /// Refuse a plan whose lookup could not fire correctly. `kernel` is the whole launched space,
    /// the only one that names the index axes (an indirect operand does not span them);
    /// `operand` is its projection onto this operand's own axes, and `projection` that operand's
    /// mapping.
    ///
    /// Every rule here is also an in-kernel assumption, but a kernel-side assert fires on a
    /// worker thread, where it reads as zeroed output rather than as a rejection, so this is the
    /// one gate. The rules: the operand carries the target axis and does not carry it innermost
    /// (that window is addressed in lines, not elements); the mapping is direct and untiled, so
    /// one dim carries the target alone; replacement routing reaches a target cut wholly inside
    /// one table entry, with whole entries above and dividing cuts below; sequence-range routing
    /// reaches a level that isolates one sequence; and every index axis is distributed across
    /// cubes or planes rather than lanes (a per-lane origin would hand `dense_lines` and
    /// `window_offset` a divergent base pointer).
    ///
    /// One rule has no host-side site and is asserted in [`Indirection::advanced_base`]: that the
    /// *operation*'s region actually spans each index axis. Which operands meet at a level is not
    /// known here, and [`Region::coord`] answers `0` for an axis its space omits, so an index axis
    /// no operand of the op spans would silently read one entry for every tile.
    pub(crate) fn validate(&self, kernel: &Space, operand: &Space, projection: &Projection) {
        assert!(
            operand.contains(self.target),
            "Indirection: the operand does not span its target axis {:?}",
            self.target
        );
        let rank = projection.physical_rank();
        assert!(
            operand.position(self.target) + 1 < rank,
            "Indirection: {:?} is the operand's innermost axis, whose window is addressed \
             in lines rather than elements; a displacement there has no line to land on",
            self.target
        );
        // The whole design rests on this: an indirection is a side-channel that moves a window,
        // never a mapping that re-addresses one. A gathered dim holds the receptive field several
        // axes reach over and a storage-tiled one splits the extent across dims, so neither has a
        // single dim the displacement could land on.
        assert!(
            projection.untiled().is_direct() && !projection.is_tiled(),
            "Indirection: an indirection rides beside a direct, untiled mapping; a gathered or \
             storage-tiled operand has no single dim carrying {:?} alone",
            self.target
        );
        assert!(
            !self.index_axes.is_empty(),
            "Indirection: an indirection with no index axis reads one entry for every tile"
        );
        if self.is_sequence_range() {
            assert_eq!(
                self.index_axes.len(),
                1,
                "Indirection: a variable-length sequence has exactly one sequence axis"
            );
        }
        assert!(
            !self.index_axes.contains(&self.target),
            "Indirection: {:?} both addresses the table and is displaced by it",
            self.target
        );
        for &axis in self.index_axes.iter() {
            assert!(
                kernel.contains(axis),
                "Indirection: {axis:?} addresses the table but is not an axis of the launched \
                 space, so no region could ever carry a coordinate for it"
            );
        }

        let mut partitioner = kernel.partitioner();
        let mut fires = false;
        while !partitioner.is_final() {
            // This level still reads the table when the lookup fires here. Its child does not,
            // so distribution below it cannot make the resolved pointer lane-divergent.
            let lookup_pending = !fires;
            if !fires {
                // The one definition of the fire level, so what validation accepts and what the
                // descent actually resolves cannot drift apart.
                fires = self.fires_with(partitioner);
                // Only replacement routing constrains how the target is cut: its entries are
                // whole slices of that axis. A sequence range translates a target tile of any
                // size once one sequence is isolated, so it has nothing to say here.
                if let IndirectionMode::Replace { granularity, .. } = self.mode {
                    let edge = partitioner.edge(self.target);
                    if fires {
                        assert!(
                            granularity.is_multiple_of(edge),
                            "Indirection: a level cuts {:?} into {edge}-element windows, which do \
                             not divide the {granularity}-element table entry, so a child window \
                             would start mid-entry",
                            self.target,
                        );
                    } else {
                        assert!(
                            edge.is_multiple_of(granularity),
                            "Indirection: a level above the lookup cuts {:?} into {edge}-element \
                             windows, which are not whole {granularity}-element table entries, so \
                             a window below would straddle two",
                            self.target,
                        );
                    }
                }
            }
            if lookup_pending {
                for &axis in self.index_axes.iter() {
                    let scope = partitioner.distribution(axis).scope();
                    assert!(
                        !matches!(scope, Some(ComputeScope::Unit)),
                        "Indirection: {axis:?} addresses the table but is distributed across \
                         lanes, so each lane would resolve a different window origin while the \
                         dense and fragment reads below share one base pointer"
                    );
                }
            }
            partitioner = partitioner.next();
        }
        match self.mode {
            IndirectionMode::Replace { granularity, .. } => assert!(
                fires,
                "Indirection: no level cuts {:?} down to a {granularity}-element table entry, so \
                 the lookup never resolves",
                self.target,
            ),
            IndirectionMode::SequenceRange => assert!(
                fires,
                "Indirection: no level isolates the variable-length sequence axis {:?} to one \
                 sequence, so the lookup never resolves",
                self.index_axes[0],
            ),
        }
    }
}

/// A data-dependent displacement of one axis's window origin, resolved exactly once during the
/// descent. Rides beside [`QuantInfo`] on [`MemData`] and follows its shape: the routing table is a
/// lifetime-erased `Box<[u32]>`, and everything deciding *where* to look is comptime
/// ([`IndirectionSpec`]).
///
/// The invariant that keeps every fast path intact is that this is a side-channel, never a
/// [`Projection`]: an indirect operand stays [`direct`](Projection::direct) and untiled, so
/// `dense_lines`, the cmma matrix view and the straight-line fill all still describe it. Above the
/// fire level nothing has moved; below it the sub-tile is dense and carries no indirection.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Indirection {
    /// The routing table, lifetime-erased like every other tile buffer.
    pub(crate) table: Box<[u32]>,
    /// The offset into `table` this descent has accumulated, advanced at each
    /// [`at`](MemData::at) by the index axes' region coordinates. Accumulated rather than read
    /// off the window's origin: the operand need not span the index axes at all, which is the
    /// whole point for MoE (`weights` is indexed by the token tile it never spans).
    pub(crate) table_base: u32,
    /// One runtime stride per entry of [`IndirectionSpec::index_axes`], same order.
    pub(crate) table_strides: Coords<u32>,
    #[cube(comptime)]
    pub(crate) spec: IndirectionSpec,
}

impl IndirectionSpec {
    /// Whether the level `space` describes resolves the lookup. Replacement routing fires once
    /// the target fits in one table entry; a sequence range fires once its sequence axis has been
    /// isolated to one sequence.
    pub(crate) fn fires_at(&self, space: &Space) -> bool {
        self.fires_with(space.partitioner())
    }

    /// The fire predicate itself, over one level. Every other fire-level question in the crate
    /// routes through this, so there is one place the rule is stated per mode.
    fn fires_with(&self, partitioner: &Partitioner) -> bool {
        match self.mode {
            IndirectionMode::Replace { granularity, .. } => {
                partitioner.edge(self.target) <= granularity
            }
            IndirectionMode::SequenceRange => partitioner.edge(self.index_axes[0]) == 1,
        }
    }

    /// The first partitioner whose child can carry a resolved window. Validation proves the walk
    /// reaches one before this helper is used by launch-side table derivation or descent math;
    /// without that proof the walk runs off the final level, which carries no edges to read.
    pub(crate) fn fire_partitioner<'a>(&self, space: &'a Space) -> &'a Partitioner {
        let mut partitioner = space.partitioner();
        while !self.fires_with(partitioner) {
            partitioner = partitioner.next();
        }
        partitioner
    }

    /// Number of table entries represented by one coordinate step at `space` for `axis`.
    /// Table coordinates name the tiles at the level where the lookup fires, while a region
    /// coordinate is local to its current parent. An outer coordinate therefore has to be
    /// scaled by the number of fire-level table entries per step below it.
    fn table_entries_per_step(&self, space: &Space, axis: Axis) -> usize {
        let current_edge = space.partitioner().edge(axis);
        let partitioner = self.fire_partitioner(space);
        let fire_edge = partitioner.edge(axis);
        comptime!(assert!(
            current_edge.is_multiple_of(fire_edge),
            "Indirection: the {current_edge}-element tile on {axis:?} does not contain a whole \
             number of {fire_edge}-element fire-level tiles"
        ));
        current_edge / fire_edge
    }
}

#[cube]
impl Indirection {
    /// This indirection with its table base advanced by `region`'s index-axis coordinates, for the
    /// child of a level that does not yet resolve it.
    pub(crate) fn advance(&self, region: &Region, #[comptime] space: Space) -> Indirection {
        Indirection {
            table: unsafe { self.table.as_boxed_unchecked() },
            table_base: self.advanced_base(region, space),
            table_strides: self.table_strides.clone(),
            spec: comptime!(self.spec.clone()),
        }
    }

    /// The table offset this level reaches: the parent's, plus each index axis's local region
    /// coordinate converted to fire-level table entry units and multiplied by its tensor stride. Read
    /// off the *region*, whose space is the operation's, not off the window: an indirect operand
    /// need not span its index axes at all, which is the whole point for MoE (the weights are
    /// routed by the token tile they never span).
    fn advanced_base(&self, region: &Region, #[comptime] space: Space) -> u32 {
        let mut parts = Sequence::<u32>::new();
        parts.push(self.table_base);
        #[unroll]
        for a in 0..comptime!(self.spec.index_axes.len()) {
            let axis = comptime!(self.spec.index_axes[a]);
            let spans = region.spans(axis);
            comptime!(assert!(
                spans,
                "Indirection: {axis:?} addresses the table but this level's region does not span \
                 it, so every tile would read the same entry"
            ));
            parts.push(
                region
                    .coord(axis)
                    .fcast::<u32>()
                    .fmul(
                        comptime!(self.spec.table_entries_per_step(&space, axis) as u32).runtime(),
                    )
                    .fmul(self.table_strides.at(a)),
            );
        }
        parts.fsum(comptime!(
            (0..self.spec.index_axes.len() + 1).collect::<Vec<_>>()
        ))
    }

    /// What this level's lookup resolves to, as `(displacement, sequence_end)`. `displacement` is
    /// how far the target axis's window origin moves, in elements of that axis. `sequence_end` is
    /// the exclusive physical end the child window is bounded at, and is meaningful only under
    /// [`IndirectionMode::SequenceRange`]; replacement routing returns `0` for it and the caller
    /// leaves the inherited bound alone. Both are `0` above the level that resolves the lookup.
    ///
    /// A replacement entry is addressed by the *absolute* logical coordinate
    /// `origin[target] + index·edge`, never by the region coordinate alone: below the top level a
    /// region coordinate is a within-parent tile index, and addressing a table with one reads
    /// garbage. A sequence range reads no target coordinate at all; its pair of entries is
    /// addressed by the accumulated index-axis base alone.
    pub(crate) fn resolve(
        &self,
        region: &Region,
        origin: &Coords<i32>,
        #[comptime] space: Space,
    ) -> (i32, u32) {
        if comptime!(self.spec.fires_at(&space)) {
            match comptime!(self.spec.mode) {
                IndirectionMode::Replace { granularity, .. } => {
                    let target = comptime!(self.spec.target);
                    let edge = comptime!(space.partitioner().edge(target) as i32);
                    let granularity = comptime!(granularity as i32).runtime();
                    let t = origin.at(comptime!(space.position(target))).fadd(
                        region
                            .coord(target)
                            .fcast::<i32>()
                            .fmul(comptime!(edge).runtime()),
                    );
                    let entry = t.fdiv(granularity);
                    let base = self.advanced_base(region, space);
                    let found = self.table[base.fadd(entry.fcast::<u32>()).fcast::<usize>()];
                    (
                        found
                            .fcast::<i32>()
                            .fmul(granularity)
                            .fsub(entry.fmul(granularity)),
                        0u32.runtime(),
                    )
                }
                IndirectionMode::SequenceRange => {
                    let base = self.advanced_base(region, space);
                    // The end is the next entry, one *stride* along, which is what
                    // `advanced_base` already scaled this base by. A literal `1` would agree
                    // only for the dense table the host validates, and the host defers that
                    // check when the sequence axis is dynamic.
                    let stride = self.table_strides.at(0);
                    let start = self.table[base.fcast::<usize>()];
                    let end = self.table[base.fadd(stride).fcast::<usize>()];
                    (start.fcast::<i32>(), end)
                }
            }
        } else {
            (0i32.runtime(), 0u32.runtime())
        }
    }
}

/// One operand's data: a runtime [`TileKind`] backing store and the comptime [`Space`] it
/// projects. `T` is the element the tile serves and computes in; its physical vector width is a
/// storage detail inside the [`TileKind`], read back with [`vector_size`](Tile::vector_size).
///
/// What an operand is at the instruction is its operand.s own statement (the finest
/// [`Residence::Register`] stage); no tile carries a second copy of it, and operands that
/// disagree meet the kind-pairing panics at the instruction, which is the same way every other
/// mismatched pair is caught.
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

    /// Create a scalar, memory-free tile over a logical space, evaluated where it is read at every
    /// level. Dynamic extents are supplied by another operand when an operation is walked; a
    /// procedural tile never witnesses them.
    fn procedural_virtual(
        #[comptime] space: Space,
        recipe: VirtualRecipe<T>,
        #[comptime] stage: StagePlan,
    ) -> Self {
        Tile::<T> {
            tile_kind: TileKind::new_Procedural(ProceduralData::<T>::new_virtual(
                comptime!(space.clone()),
                recipe,
                stage,
            )),
            space,
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
            TileKind::Gmem(d) | TileKind::Smem(d) => d.lane_share,
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(LaneShare::Whole)
            }
        }
    }

    /// Ask this accumulator to start from `init_from`, and answer what actually took.
    ///
    /// Asking rather than deciding is what keeps a kind that cannot take the request from having
    /// to be listed anywhere else.
    pub(crate) fn request_init_from(
        &mut self,
        #[comptime] init_from: InitFrom,
    ) -> comptime_type!(InitFrom) {
        match &mut self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => {
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

    /// Whether this tile's buffer is addressed through a mapping whose windows may *overlap*: a
    /// physical cell does not determine the logical position, so the only read surface that
    /// describes the tile is the N-D one ([`nd`](Tile::nd)) and no window of it is dense.
    ///
    /// False for a [`direct`](Projection::direct) operand, and equally for one whose axes
    /// [partition](Composition::Disjoint) a physical axis: a partition is a bijection, so its
    /// windows tile rather than overlap and every dense path still describes it. A fragment or a
    /// tensor map has no buffer to gather from.
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
    /// that factor reads matters, which is what lets the contraction walk it in 1-D.
    ///
    /// Asking [`factors`](Tile::factors) first is the whole precondition: a tile answering `None`
    /// has no factorization to index into, and is exactly the tile kind that cannot evaluate one.
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
    /// The axes of `walk` this tile is constant along, so that one read of it serves every
    /// position of all of them at once.
    ///
    /// Two spellings, one fact. An axis the tile's own space does not span it cannot vary over at
    /// all; an axis it spans but whose projection addresses nothing it cannot vary over either.
    /// Both say the same thing — the operand distinguishes nothing there — and a consumer that
    /// wants to know how far out of a nest a read lifts wants both.
    ///
    /// The [gathered nest](crate::FactorReuse) asks this of a procedural factor through its
    /// recipe, which reads its own axes rather than addressing them. Same question, and the answer
    /// means the same thing: the scope over which the value holds.
    pub(crate) fn invariant_over(&self, #[comptime] walk: Space) -> comptime_type!(Vec<Axis>) {
        let projection = self.projection();
        comptime!(
            walk.axes()
                .filter(|axis| !self.space.contains(*axis) || !projection.addresses(*axis))
                .collect::<Vec<_>>()
        )
    }

    pub fn witnesses(&self, #[comptime] axis: Axis) -> comptime_type!(bool) {
        let bounded = self.bounded();
        let projection = self.projection();
        // An axis an indirection targets is the fourth way an operand spans one without being
        // able to answer for it: its bound is the index tensor's range (a whole KV cache, every
        // expert's weights), never the operation's extent along that axis.
        let indirect = self.indirection_target();
        comptime!(
            bounded
                && indirect != Some(axis)
                && self.space.contains(axis)
                && self.space.is_dynamic(axis)
                && bound_states(&projection, axis).is_some()
        )
    }

    /// Which axes' region coordinates address this operand's routing table; empty for an operand
    /// with no indirection. Comptime, and read by the staging plan
    /// ([`Space::walk_invariant`](crate::Space::walk_invariant)).
    pub(crate) fn index_axes(&self) -> comptime_type!(SmallVec<[Axis; MAX_AXES]>) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.index_axes(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(SmallVec::new())
            }
        }
    }

    /// The axis an indirection displaces on this operand, if it carries one.
    pub(crate) fn indirection_target(&self) -> comptime_type!(Option<Axis>) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.indirection_target(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(None::<Axis>)
            }
        }
    }

    /// The indirection spec if this operand carries a pending indirection, empty otherwise.
    pub(crate) fn indirection_spec(&self) -> comptime_type!(Option<IndirectionSpec>) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.indirection_spec(),
            TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_)
            | TileKind::Procedural(_) => {
                comptime!(None::<IndirectionSpec>)
            }
        }
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

    /// Seed this tile with `monoid`'s identity, so a fold under it starts from a value folding
    /// it in leaves unchanged.
    ///
    /// `Sum` goes through [`zero`](Tile::zero), which every accumulator form can do, hardware mma
    /// fragments included; the other monoids need a real value and so reach only the forms
    /// [`init`](Tile::init) serves.
    pub fn init_identity(&mut self, #[comptime] monoid: Monoid) {
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

    /// Drain a resident accumulator into memory `dst`, casting `T` down to `dst`'s element
    /// type. [`copy_from`](Self::copy_from) can't: its transports move bytes so stay same-type,
    /// but a register accumulator (`f32`) is wider than the output it writes (`f16`). Only a
    /// fragment partition drains this way.
    ///
    /// Crate-internal: what closes an accumulator's scope is the scope's own business
    /// ([`AccumulatorScope`](crate::AccumulatorScope)), not a call site's.
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
