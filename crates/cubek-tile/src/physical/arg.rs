//! The launch-side operand vocabulary: the [`Projection`] mapping, the comptime [`TileSpec`]
//! a kernel feeds [`Tile::of`](crate::Tile::of), and the TMA argument [`TmaTileArg`]
//! (a tensor map cannot ride a plain tensor binding, so it keeps its own carrier).

use cubecl::prelude::*;
use cubecl::quant::scheme::{QuantScheme, QuantStore, ScaleDtype};
use cubecl::std::quant::view::KnownScale;
use cubecl::std::tensor::{
    ViewMut,
    layout::{CoordsDyn, Layout, LayoutExpand, linear::LinearView},
    view::launch::ViewArg,
};
use cubecl::zspace::SmallVec;

use crate::*;

/// The comptime half of an operand: which axes of the kernel's one [`Space`] its buffer
/// spans and how they address its physical axes ([`Projection`], which carries the buffer's
/// storage tiling in its own repetition). What a kernel feeds [`Tile::of`](crate::Tile::of)
/// alongside that space; `of` projects the space onto the projection's logical axes, so no operand
/// ever carries its own copy of the space. The launch-side builder derives it
/// ([`build`](crate::StridedTileSource::build)).
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TileSpec {
    /// How this operand's logical axes address its buffer's physical ones.
    pub projection: Projection,
    /// How out-of-bounds reads/writes are handled (`None` keeps the unchecked divisible fast path;
    /// `Some(Boundary::Zero)` returns zero on reads / skips writes; `Some(Boundary::Clamp)` clamps).
    pub boundary: Option<Boundary>,
    /// The launch's cube size (units per cube), `0` when unknown; carried into the
    /// [`StagePlan`](crate::StagePlan) of every stage derived from this operand.
    pub units: usize,
    /// Explicit stage-layout override; `None` derives from this operand's leaf in
    /// [`Tile::of`](crate::Tile::of) ([`StageStorage::for_leaf`]).
    pub storage: Option<StageStorage>,
    /// Where this operand lives at each level of the kernel's space, coarse to fine. Empty (the
    /// default) is every level [`InPlace`](Residence::InPlace): read where it already is, staging
    /// nothing.
    pub residence: SmallVec<[Residence; MAX_LEVELS]>,
    /// What this operand is at the instruction: a memory window, or a plane fragment in one of the
    /// two encodings. Stated when the spec is built, so no operand ever carries an implied one; a
    /// format decision, so it belongs to the operand rather than to the partitioning.
    /// [`Tile::of`](crate::Tile::of) carries it onto the tile, and operands that disagree meet the
    /// kind-pairing panics at the instruction.
    pub leaf: Leaf,
}

impl TileSpec {
    /// An operand's spec from its mapping and its leaf; the optional halves are the safe defaults
    /// (unchecked, cube size unknown, nothing staged) and are set by
    /// [`boundary`](Self::boundary), [`checked`](Self::checked), [`units`](Self::units),
    /// [`storage`](Self::storage), and [`residence`](Self::residence).
    ///
    /// [`Projection::validate`] is not run here: its innermost-identity rule turns on the served
    /// vector width, which a spec does not carry and only [`Tile::of`](crate::Tile::of) knows, so
    /// that is where it runs instead.
    pub fn new(projection: Projection, leaf: Leaf) -> Self {
        TileSpec {
            projection,
            boundary: None,
            units: 0,
            storage: None,
            residence: SmallVec::new(),
            leaf,
        }
    }

    /// [`new`](Self::new) over the [`direct`](Projection::direct) mapping: one logical axis per
    /// physical axis, which is every non-gather, untiled operand.
    pub fn direct(axes: &[Axis], leaf: Leaf) -> Self {
        Self::new(Projection::direct(axes), leaf)
    }

    /// The axes this operand spans, in its logical order.
    pub fn axes(&self) -> &[Axis] {
        self.projection.logical_axes()
    }

    /// Override the derived stages' [`StageStorage`] layout (default
    /// [`StageStorage::for_leaf`]).
    pub fn storage(mut self, layout: StageStorage) -> Self {
        self.storage = Some(layout);
        self
    }

    /// Where this operand lives at each level of the kernel's space, coarse to fine (default: every
    /// level [`InPlace`](Residence::InPlace)). Length-checked against the space by
    /// [`StridedTileSource::build`](crate::StridedTileSource::build), which is the only builder
    /// that has one; a hand-written spec is trusted.
    pub fn residence(mut self, residence: &[Residence]) -> Self {
        self.residence = SmallVec::from_slice(residence);
        self
    }

    /// This operand's [`StagePlan`]: its per-level residences, plus the stage layout its
    /// materialized levels take (the explicit override, else derived from its leaf).
    pub fn stage_plan(&self) -> StagePlan {
        StagePlan::new(
            &self.residence,
            self.storage
                .unwrap_or_else(|| StageStorage::for_leaf(self.leaf)),
            self.units,
        )
    }

    /// Set whether edge reads/writes must be bounds-checked with [`Boundary::Zero`]. A boolean
    /// convenience over [`with_boundary`](Self::with_boundary): it unconditionally overwrites
    /// whatever mode was set before it, so a `with_boundary(Some(Boundary::Clamp))` before this
    /// call is silently dropped back to `Zero`. Sequence a `Clamp` override after `checked`, not
    /// before it.
    pub fn checked(mut self, check: bool) -> Self {
        self.boundary = if check { Some(Boundary::Zero) } else { None };
        self
    }

    /// Set the optional boundary handling mode for out-of-bounds access.
    pub fn with_boundary(mut self, boundary: Option<Boundary>) -> Self {
        self.boundary = boundary;
        self
    }

    /// Whether this operand has bounds checking enabled.
    pub fn is_checked(&self) -> bool {
        self.boundary.is_some()
    }

    /// Set the launch's cube size (units per cube).
    pub fn units(mut self, units: usize) -> Self {
        self.units = units;
        self
    }
}

/// One strided operand as a single launch argument: the plain tensor (whose element type
/// carries the served width) paired with its comptime [`TileSpec`], so a tensor can never
/// be launched against another operand's spec. Only per-operand facts live here; the
/// kernel's one [`Space`] arrives separately and [`tile`](TileArg::tile) projects it.
#[derive(CubeType, CubeLaunch)]
pub struct TileArg<'a, E: Numeric, V: Size> {
    pub tensor: &'a Tensor<Vector<E, V>>,
    #[cube(comptime)]
    pub spec: TileSpec,
}

#[cube]
impl<'a, E: Numeric, V: Size> TileArg<'a, E, V> {
    /// Serve the operand as a [`Tile`]: the kernel's one `space` projected onto this
    /// operand's `spec` axes.
    pub fn tile(&self, #[comptime] space: Space) -> Tile<E> {
        Tile::<E>::of(self.tensor, space, comptime!(self.spec.clone()))
    }

    /// [`tile`](Self::tile) for a gather whose affine map is not all comptime: `coefficients` holds
    /// one value per [`Scale::Dynamic`](crate::Scale) term and one per
    /// [`Divisor::Dynamic`](crate::Divisor) axis of the spec's projection, `offsets` one signed
    /// value per [`Offset::Dynamic`](crate::Offset) axis. See [`Tile::of_gathered`] for the order
    /// each carrier is read in, which only it enforces.
    pub fn tile_gathered(
        &self,
        #[comptime] space: Space,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
    ) -> Tile<E> {
        Tile::<E>::of_gathered(
            self.tensor,
            space,
            comptime!(self.spec.clone()),
            coefficients,
            offsets,
        )
    }
}

/// One quantized operand as a single launch argument: the storage-typed values tensor
/// (`E` is the *stored* scalar: `u32` words packed, `i8` native), its scales, and the
/// comptime spec + scheme. A quantized tensor is one thing, so its pieces travel as one
/// argument; [`TileArg`] is its plain twin. Only per-operand facts live here; the
/// kernel's one [`Space`] arrives separately and [`tile`](QuantTileArg::tile) projects it.
#[derive(CubeType, CubeLaunch)]
pub struct QuantTileArg<'a, E: Numeric, V: Size> {
    pub values: &'a Tensor<Vector<E, V>>,
    pub scales: &'a Tensor<f32>,
    /// The global level's whole-tensor scale in its first element, bound exactly when the scheme
    /// has a second level.
    pub global: ComptimeOption<LinearView<'static, f32>>,
    /// A lookup scheme's `2^bits`-entry table, present exactly under
    /// [`QuantMode::Lookup`](cubecl::quant::scheme::QuantMode): reads reconstruct
    /// `table[field] * scale` instead of casting the field.
    pub table: ComptimeOption<Box<[f32]>>,
    #[cube(comptime)]
    pub spec: TileSpec,
    #[cube(comptime)]
    pub scheme: QuantScheme,
    /// How far this operand stays quantized; stated at launch, where what can decode is known.
    #[cube(comptime)]
    pub dequant_at: DequantAt,
}

#[cube]
impl<'a, E: Numeric, V: Size> QuantTileArg<'a, E, V> {
    /// Serve the operand as a [`Tile`] of the served type `O`: the kernel's one `space`
    /// projected onto this operand's `spec` axes, reads dequantizing per the scheme.
    pub fn tile<O: Numeric>(&self, #[comptime] space: Space) -> Tile<O> {
        // The engine's own backstop, like `validate_dequant_at`: the builder checks the binding
        // contract too, but a hand-built `QuantTileArgLaunch` reaches here without it.
        comptime!(cubecl::std::quant::check_scale_bindings(
            &self.scheme,
            1 + self.global.is_some() as usize
        ));
        // One read for the whole kernel; every window below shares the register.
        let known = if comptime!(self.global.is_some()) {
            let buffer = self.global.unwrap();
            KnownScale::new_Global(buffer.read(0))
        } else {
            KnownScale::new_None()
        };
        Tile::<O>::of_dequant(
            self.values,
            self.scales,
            known,
            self.table.clone(),
            comptime!(self.scheme),
            comptime!(self.dequant_at),
            space,
            comptime!(self.spec.clone()),
        )
    }
}

/// The TMA [`Delivery`]'s argument: the tensor-map [`ViewMut`] carrier (the descriptor
/// owns the box, the [`TmaDynLayout`] the coordinate rules) paired with its comptime
/// [`TileSpec`], [`TileArg`]'s twin (a tensor map cannot ride a plain tensor binding).
/// Built by [`TmaTileArgLaunch::tensor_map`](crate::TmaTileArgLaunch::tensor_map).
#[derive(CubeType, CubeLaunch)]
pub struct TmaTileArg<E: Numeric> {
    pub view: ViewMut<'static, E, CoordsDyn>,
    #[cube(comptime)]
    pub spec: TileSpec,
}

#[cube]
impl<E: Numeric> TmaTileArg<E> {
    /// Serve the tensor map as a [`TmaGmem`](crate::TileKind::TmaGmem) tile over the
    /// kernel's one `space`; the spec's width and storage don't apply to a tensor map.
    pub fn tile(&self, #[comptime] space: Space) -> Tile<E> {
        TmaData::from_tensor_map(
            self.view.clone(),
            comptime!(space.project(self.spec.axes())),
            comptime!(self.spec.leaf),
            comptime!(self.spec.stage_plan()),
        )
    }
}

/// Reject a [`QuantScheme`] this operand cannot serve, at launch and on the caller's thread. Every
/// rule here is also an in-kernel assumption, but a kernel-side assert fires on a device thread,
/// where it reads as zeroed output rather than as a rejection, so this is the one gate.
///
/// A tile reads a scale as its window's own start plus the block index *within* the window
/// ([`ScaleLayout`]), which is the true block only if no window straddles a block edge. Every
/// window is a level's cut, and its origin is a multiple of that cut, so per axis each level's
/// edge must tile whole blocks or fit inside one. A line is one read, so it may not straddle
/// either. A per-tensor scale covers every window and line, so only the store and param rules
/// apply to it.
pub(crate) fn validate_scheme(space: &Space, vector_size: usize, scheme: QuantScheme) {
    // `Native` holds one element per value; `PackedU32` carries `num_quants` of them per `u32`,
    // which the view unpacks on read. A packed store must pack along the innermost (contiguous,
    // vectorized) axis, the one whose lanes the view lays down contiguously. Sub-byte
    // native stores aren't wired.
    match scheme.store {
        QuantStore::Native => {}
        QuantStore::PackedU32(dim) => {
            assert!(
                dim == 0,
                "StridedTileSource::quantized: a packed-u32 operand must pack along the \
                 innermost axis (dim 0), got {dim}"
            );
            assert!(
                vector_size.is_multiple_of(scheme.num_quants()),
                "StridedTileSource::quantized: the innermost axis is served in \
                 {vector_size}-wide lines, which must be a multiple of the {}-value packing \
                 factor, else a line splits a u32",
                scheme.num_quants()
            );
        }
        other => panic!(
            "StridedTileSource::quantized: quantization storage {other:?} is not supported \
             (native or packed-u32)"
        ),
    }
    // The scales ride a plain `f32` tensor read straight through, so a narrower param
    // would reinterpret its bytes.
    assert!(
        scheme.scale_dtype() == ScaleDtype::F32,
        "StridedTileSource::quantized: scales are read as f32, got {:?}",
        scheme.scale_dtype()
    );

    // A per-tensor scale has no edge to straddle, whatever the cuts and the served width.
    if scheme.block_size().is_none() {
        return;
    }

    let rank = space.rank();
    let block = block_edges(scheme, rank);
    let inner = block[rank - 1];
    assert!(
        inner.is_multiple_of(vector_size),
        "StridedTileSource::quantized: the innermost axis is served in {vector_size}-wide \
         lines, which its {inner}-element scale blocks must be a multiple of, else one line \
         straddles two scales"
    );

    // Every window is some level's cut, so the final space (which carries no cut) has nothing
    // left to check: its extents are the last level's edges.
    let mut level = space.clone();
    while !level.is_final() {
        for (p, axis) in level.axes().enumerate() {
            let (edge, block) = (level.partitioner().edge(axis), block[p]);
            assert!(
                edge.is_multiple_of(block) || block.is_multiple_of(edge),
                "StridedTileSource::quantized: {axis:?} is cut into {edge}-element tiles, \
                 which straddle its {block}-element scale blocks; a tile must cover whole blocks \
                 or sit inside one"
            );
        }
        level = level.divide();
    }
}

impl<E: Numeric, R: Runtime> TmaTileArgLaunch<E, R> {
    /// Load a TMA tensor-map as a tile argument. `dims` is the operand's logical runtime
    /// `(batch, rows, cols)`; `transposed` flags a col-major operand whose descriptor
    /// swapped its inner pair (the layout swaps coords back). `axes` are the operand's
    /// spanned axes: 3 with a leading batch axis, 2 without. The spec's width and storage
    /// don't apply to a tensor map, so the spec is built here, not by the caller.
    pub fn tensor_map(
        tensor_map: TensorMapArg<R, Tiled>,
        axes: &[Axis],
        dims: (u32, u32, u32),
        transposed: bool,
        leaf: Leaf,
        residence: &[Residence],
    ) -> Self {
        let batched = match axes.len() {
            2 => false,
            3 => true,
            r => panic!(
                "TmaTileArg: the descriptor is (batch, row, col); rank {r} operand unsupported"
            ),
        };
        let layout = TmaDynLayoutLaunch::new(dims, batched, transposed);
        let view = ViewArg::new_tensor_map_tiled::<TmaDynLayout>(tensor_map, layout);
        Self::new(view, TileSpec::direct(axes, leaf).residence(residence))
    }
}

/// In-kernel tensor-map layout: aligns the operand's logical [`CoordsDyn`] to the
/// descriptor's 3-D `(batch, row, col)`. A batchless (rank-2) operand gets batch `0`, a
/// unit batch broadcasts; a col-major (`transposed`) descriptor swapped its inner pair, so
/// the layout swaps coords back. `shape()` stays logical, so a tile's `bound` aligns with
/// its space whatever the descriptor order. Same rules as cubek-matmul's legacy
/// `SimpleTmaGlobalLayout` (tuple coords); keep the two in step.
#[derive(CubeType, CubeLaunch, Clone)]
pub struct TmaDynLayout {
    /// Logical `(batch, rows, cols)` of the operand.
    dims: (u32, u32, u32),
    #[cube(comptime)]
    batched: bool,
    #[cube(comptime)]
    transposed: bool,
}

#[cube]
impl Layout for TmaDynLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (batch, _rows, _cols) = self.dims;
        let mut src = CoordsDyn::new();
        if comptime!(self.batched) {
            // A unit-batch descriptor is a broadcast: always read batch 0.
            src.push(select(batch == 1, 0u32, pos[0]));
        } else {
            src.push(0u32);
        }
        let (r, c) = comptime!(if self.batched { (1, 2) } else { (0, 1) });
        // TMA discards the last stride, so a col-major descriptor is transposed; swap back.
        if comptime!(self.transposed) {
            src.push(pos[c]);
            src.push(pos[r]);
        } else {
            src.push(pos[r]);
            src.push(pos[c]);
        }
        src
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        // TMA loads are clamped by the descriptor; no in-kernel bounds check.
        (self.to_source_pos(pos), true)
    }

    fn shape(&self) -> Self::Coordinates {
        let (batch, rows, cols) = self.dims;
        let mut s = CoordsDyn::new();
        if comptime!(self.batched) {
            s.push(batch);
        }
        s.push(rows);
        s.push(cols);
        s
    }

    fn is_in_bounds(&self, _pos: Self::Coordinates) -> bool {
        true
    }
}
