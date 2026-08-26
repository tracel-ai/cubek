//! The addressable backing store ([`MemData`], gmem and smem): its layouts, windows, and
//! the memory-side [`Tile`] operations (construction, views, the cooperative copy).

use cubecl::zspace::SmallVec;
use cubecl::{
    prelude::*,
    quant::scheme::{QuantScheme, QuantStore, QuantValue},
    std::quant::unpack_fields,
    std::quant::view::KnownScale,
    std::tensor::{
        AsView, AsViewExpand, AsViewMut, AsViewMutExpand, View, ViewMut,
        layout::{Coordinates, Coords1d, Coords2d, CoordsDyn, Layout, LayoutExpand},
    },
};

use crate::*;

/// A lifetime-erased buffer, how to address it ([`layout`](GmemLayout)), and which part of it this
/// tile is looking at ([`window`](Window)). The layout is fixed at construction, so a staged smem
/// sub-tile keeps addressing its whole buffer after [`at`](Tile::at) windows it down.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct MemData<T: Numeric> {
    /// What the bytes are and mean.
    pub(crate) store: Store<T>,
    /// How a logical coordinate becomes a buffer offset. Fixed at construction.
    pub(crate) layout: GmemLayout,
    /// The region of the *physical* buffer this tile covers; narrowed by [`at`](Tile::at).
    pub(crate) window: Window,
    /// How the tile's logical axes address the buffer's physical ones.
    /// [`direct`](Projection::direct) for every non-gather operand, where logical rank equals
    /// physical rank and this is the identity;
    /// an affine map for an operand gathered over an abstract dimension. Fixed at construction,
    /// like the layout: `at` moves the window, never the mapping.
    #[cube(comptime)]
    pub(crate) projection: Projection,
    /// What [`projection`](Self::projection) only knows in the kernel: its runtime coefficients and
    /// the phase its window origin sits at. [`integral`](RuntimeMap::integral) for every operand
    /// but a runtime-strided or fractionally scaled gather.
    pub(crate) map: RuntimeMap,
    /// The runtime half of the projection's constant terms: one value per
    /// [`Offset::Dynamic`](crate::Offset) axis, in [`Projection::dynamic_offset_index`] order.
    /// Signed, since a padding places the window before the buffer's origin. Not part of
    /// [`map`](Self::map): an offset only ever places the top window, which
    /// [`window`](Self::window) then carries, so nothing below reads it again.
    pub(crate) offsets: Coords<i32>,
    /// The window origin's offset through the layout, accumulated across [`at`](Tile::at)s rather
    /// than re-derived: each descent shifts by a *comptime* edge, so [`step_offset`] folds
    /// and this stays a multiply-add. Addressing it from the origin instead would decompose a
    /// runtime coordinate, i.e. integer division per [`window_slice`](MemData::window_slice).
    window_start: u32,
    /// How this store may be touched. All comptime, all decided at construction.
    #[cube(comptime)]
    pub(crate) access: Access,
    /// What each lane holds of these cells, stamped across [`at`](Tile::at)s (the level that
    /// spreads an axis is consumed on the way down). `Partial` means split to an accumulator but
    /// merely replicated to an operand, so only an accumulator reads it.
    #[cube(comptime)]
    pub(crate) lane_share: LaneShare,
    /// What the accumulation being lowered right now starts from ([`InitFrom`]).
    ///
    /// Not a claim about the bytes: it says nothing about what the buffer holds, only what the
    /// caller asked for. [`Tile::mm`] and [`Tile::reduce_axis`] state
    /// [`Identity`](InitFrom::Identity) for the span of their own lowering, having proven the leaf
    /// visits each cell once ([`Space::spans_contracted_at_leaf`]); it is
    /// [`Cell`](InitFrom::Cell) everywhere else, and rides [`at`](MemData::at) down so the levels
    /// below see the verb the caller used.
    #[cube(comptime)]
    pub(crate) init_from: InitFrom,
    /// Where this tile's cells sit inside the buffer they were *filled from*, when that is not
    /// the buffer they live in.
    ///
    /// `None` for every tile that reads its source directly: there [`window`](Self::window)
    /// already is the source window, so a boundary question is answered against it. `Some` only
    /// for a gathered stage, whose fill replaced out-of-bounds samples with the boundary's value
    /// and whose own window can no longer say which those were.
    pub(crate) source_window: ComptimeOption<SourceWindow>,
}

/// What a [`MemData`]'s bytes are and mean: the erased buffer, the width it groups into lines at,
/// and, when the buffer physically holds quantized data, how a *stored* value becomes a *served*
/// one. Reads through [`Tile::flat`] dequantize into `T`; every other element view refuses a
/// quantized tile.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Store<T: Numeric> {
    /// Backing store, scalar-typed by Rust-side erasure only: the real binding/alloc element is
    /// `Vector<T, vector_size>`, so re-grouping to lines at that width is a no-op.
    pub(crate) buffer: Box<[T]>,
    /// Physical line size (`Vector<T, vector_size>`) of the backing store, `1` when
    /// unvectorized; held comptime so `size!` can read it.
    #[cube(comptime)]
    pub(crate) vector_size: usize,
    /// Present when the buffer holds quantized data (see [`QuantInfo`]).
    pub(crate) quant: ComptimeOption<QuantInfo>,
    /// How the buffer's values sit in it: whether a stored element *is* a served one, and what a
    /// read has to unpack if it is not. Stated at construction, from the operand's spec
    /// ([`TileSpec::packed`]) or from its scheme where it has one, so no reader re-derives it.
    #[cube(comptime)]
    pub(crate) packing: Packing,
}

/// How a [`MemData`] may be touched: whether the fill can write straight through, how the store
/// handles overhang, and how a cooperative fill spreads. Plain data held comptime, like the
/// [`StagePlan`] it carries.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Access {
    /// Whether the window still covers the whole buffer (constructors yes, [`at`](Tile::at) no):
    /// such a tile can be written in physical order.
    pub whole: bool,
    pub overhang: Overhang,
    /// Where this operand lives at each level below, plus the [`StageStorage`] layout and launch
    /// cube size its materialized levels take. Carried from the operand's [`TileSpec`] so a fill
    /// re-derives none of them.
    pub stage: StagePlan,
}

/// How a store relates to the window overhanging its valid data (`origin + pos` past
/// [`Window`]'s `bound`); where gmem and smem genuinely differ.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Overhang {
    /// Structurally impossible: the buffer is allocated to exactly the tile (smem).
    Never,
    /// Possible in principle, excluded at launch: every shape divides its tiling (unchecked gmem).
    Fits,
    /// Possible: reads/writes past `bound` are masked, per the window's [`Boundary`] (zero for
    /// reads and skipped for writes under `Zero`, the edge cell under `Clamp`).
    Masked,
}

/// Boundary handling mode for out-of-bounds reads/writes, carried by [`Window`] (the layer that
/// owns `origin`/`bound`/`signed` and so is the one that can turn an out-of-range coordinate into
/// a valid physical one).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Boundary {
    /// Out-of-bounds reads return zero; writes are skipped.
    Zero,
    /// Out-of-bounds reads/writes clamp to the edge cell.
    Clamp,
}

impl Overhang {
    /// The flag a [`MaskedView`] is built with; the one place the states collapse to a bool.
    pub fn masks(&self) -> bool {
        matches!(self, Overhang::Masked)
    }
}

/// Whether a read still proves its own bounds, stated by the reader rather than read off the
/// tile. Comptime, so the arm not taken costs nothing.
///
/// A tile records what it *could* need ([`Overhang`], the window's [`Boundary`]); this records
/// what a particular reader has established it needs, which is the weaker claim and the only one
/// a leaf splitting itself across an edge can make.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Guard {
    /// Mask the overhang and apply the window's [`Boundary`] on every access.
    Checked,
    /// The reader has proved the whole box it will touch lands inside the buffer, so the view
    /// carries neither. Reading through it past that box is out of bounds, not masked.
    Proved,
}

impl Guard {
    /// Whether this guard still costs a test per access.
    pub fn checks(self) -> bool {
        matches!(self, Guard::Checked)
    }
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// Construct a whole `Gmem` tile straight from a launched tensor: the kernel's one
    /// `space` projected onto the operand's `spec` axes, so no operand carries its own
    /// copy of the space. The element type carries the line width: `Vector<T, W>` for a
    /// lined operand, `T` itself for scalar, so the served width *is* the binding's
    /// width by construction and is never re-lined in-kernel. Shape/strides come in
    /// scalar-unit and convert to line-unit here.
    pub fn of<E: CubePrimitive<Scalar = T>>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> Tile<T> {
        Tile::<T>::of_impl::<E>(
            tensor,
            space,
            spec,
            ComptimeOption::new_None(),
            Coords::<u32>::new(),
            Coords::<i32>::new(),
        )
    }

    /// [`of`](Tile::of) for a gather whose affine map is not all comptime: `coefficients` holds one
    /// value per [`Scale::Dynamic`](crate::Scale) term in [`Projection::dynamic_scale_index`] order
    /// *and* one per [`Divisor::Dynamic`](crate::Divisor) axis in
    /// [`Projection::dynamic_divisor_index`] order, the two interleaved physical axis major so an
    /// axis's divisor follows its own coefficients; `offsets` one signed value per
    /// [`Offset::Dynamic`](crate::Offset) axis in [`Projection::dynamic_offset_index`] order. A
    /// runtime stride, dilation, padding or resize ratio is exactly this, and the kernel builds the
    /// carriers from its own scalar arguments, so nothing about them reaches the launch.
    ///
    /// Only the lengths are checked, so the two index orders above are the contract: swap a
    /// coefficient for a divisor and the read is silently wrong.
    pub fn of_gathered<E: CubePrimitive<Scalar = T>>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
    ) -> Tile<T> {
        Tile::<T>::of_impl::<E>(
            tensor,
            space,
            spec,
            ComptimeOption::new_None(),
            coefficients,
            offsets,
        )
    }

    /// [`of`](Tile::of) from a [`packed`](TileSpec::packed) operand: the binding holds stored
    /// `u32` words and the tile serves the values inside them, `factor` per word, unpacked at the
    /// read. The served width is the binding's width × that factor.
    ///
    /// No scales and no scheme: what the spec states is a fact about *these* values, and an
    /// operand that also has scales names them as its own tensor.
    pub fn of_packed<E: CubePrimitive>(
        values: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> Tile<T> {
        comptime!(assert!(
            spec.packing != Packing::Plain,
            "Tile::of_packed: the operand states no packing, so it is a plain tile (Tile::of)"
        ));
        Tile::<T>::of_impl::<E>(
            values,
            space,
            spec,
            ComptimeOption::new_None(),
            Coords::<u32>::new(),
            Coords::<i32>::new(),
        )
    }

    /// [`of`](Tile::of) from a quantized operand: the values tensor is storage-typed (its
    /// element's scalar is the *stored* type: `u32` words for a packed scheme, `i8` native),
    /// the scales ride as a plain second tensor, and the comptime scheme says how reads fold
    /// them back in. The served width is the binding's width × the scheme's packing factor.
    #[allow(clippy::too_many_arguments)]
    pub fn of_dequant<E: CubePrimitive>(
        values: &Tensor<E>,
        scales: &Tensor<f32>,
        known: KnownScale,
        table: ComptimeOption<Box<[f32]>>,
        #[comptime] scheme: QuantScheme,
        #[comptime] dequant_at: DequantAt,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> Tile<T> {
        // The engine's own backstop: the builder checks this too, but a hand-built
        // `QuantTileArgLaunch` reaches here without passing through it.
        comptime!(validate_dequant_at(dequant_at, space.instruction()));
        comptime!(cubecl::std::quant::check_table_bindings(
            &scheme,
            table.is_some()
        ));
        let rank = comptime!(spec.axes().len());
        let block = comptime!(block_edges(scheme, rank));
        let mut strides = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            if comptime!(scheme.block_size().is_none()) {
                strides.push(0u32);
            } else {
                strides.push(scales.stride(p) as u32);
            }
        }
        let info = QuantInfo {
            buffer: unsafe { scales.as_slice().as_boxed_unchecked() },
            known,
            strides,
            window_start: 0u32,
            block: comptime!(block),
            extent: comptime!(window_extents(&space.project(spec.axes()), rank)),
            dequant_at: comptime!(dequant_at),
            // A gmem operand reads the tensor's scales in place; only a staged stage grids them.
            scale_shape: comptime!(Vec::new()),
            table,
            scheme: comptime!(scheme),
        };
        Tile::<T>::of_impl::<E>(
            values,
            space,
            spec,
            ComptimeOption::new_Some(info),
            Coords::<u32>::new(),
            Coords::<i32>::new(),
        )
    }

    /// Shared body of [`of`](Tile::of)/[`of_dequant`](Tile::of_dequant): `E` is the *binding*
    /// element (its scalar the stored type), `T` the served scalar; they differ only for a
    /// quantized operand, whose served width is the binding's width × the packing factor.
    fn of_impl<E: CubePrimitive>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
        quant: ComptimeOption<QuantInfo>,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
    ) -> Tile<T> {
        // The one projection: the kernel's space narrowed to this operand's axes.
        let space = comptime!(space.project(spec.axes()));
        let projection = comptime!(spec.projection.clone());
        // The operand addresses *coordinates*; the buffer's storage tiling is the layout's business
        // ([`positional`] below), and splitting a coordinate into digits is what it does with it.
        let coords = comptime!(projection.untiled());
        // The scales are gridded over the operand's *logical* axes (`strides` and `block` below
        // are one entry per logical axis), while `at` re-windows them over the physical ones. The
        // two ranks coincide for every direct operand, tiled or not, and diverge under a gather.
        comptime!(assert!(
            quant.is_none() || coords.is_direct(),
            "Tile::of: a gathered operand cannot be quantized; its scale grid is shaped over its \
             logical axes, which its buffer's physical axes no longer match"
        ));
        let scales_given = coefficients.len();
        comptime!(assert!(
            scales_given == coords.dynamic_coefficient_count(),
            "Tile::of: the projection has {} Dynamic coefficients and divisors but \
             {scales_given} were given",
            coords.dynamic_coefficient_count()
        ));
        let offsets_given = offsets.len();
        comptime!(assert!(
            offsets_given == coords.dynamic_offset_count(),
            "Tile::of: the projection has {} Dynamic offsets but {offsets_given} were given",
            coords.dynamic_offset_count()
        ));
        let stage = comptime!(spec.stage_plan(space.instruction()));
        // How the buffer holds its values: what a quantized operand's scheme says, else what the
        // spec states. One statement, whichever door minted it, so nothing below asks twice.
        let packing = #[comptime]
        match &quant {
            ComptimeOption::Some(info) => comptime!(scheme_packing(info.scheme)),
            ComptimeOption::None => comptime!(spec.packing),
        };
        comptime!(assert!(
            quant.is_none() || spec.packing == Packing::Plain,
            "Tile::of: a quantized operand's scheme already states how its values are stored, so \
             its spec may not state a packing too"
        ));
        // The binding type's own width, comptime; a packed store serves `factor` values per
        // stored element.
        let bound_width = tensor.vector_size();
        let vector_size = comptime!(bound_width * packing.factor());
        // The operand's own contract, checked here rather than at `TileSpec` construction because
        // it turns on the served width, which only this call, not the spec, ever knows. Same for a
        // padded stage width, which `StridedTileSource` already checked for the specs it builds;
        // this catches hand-built ones too.
        comptime!(projection.validate(vector_size));
        // A `Disjoint` claim is about the axes' extents, and this is the one place the projection
        // and the space are both in hand.
        comptime!(projection.validate_composition(|axis| space.extent(axis)));
        comptime!(spec.validate_stage_width(vector_size, packing != Packing::Plain));
        let coord_rank = comptime!(projection.coordinate_rank());
        comptime!(assert!(
            spec.boundaries.is_empty() || spec.boundaries.len() == coord_rank,
            "Tile::of: boundaries rank ({}) does not match coordinate rank ({coord_rank})",
            spec.boundaries.len()
        ));
        // A clamped vector line is only valid if the innermost coordinate axis is not clamped. The
        // source builder derives that per-axis mask; this catches hand-built specs too.
        comptime!(assert!(
            vector_size == 1 || spec.boundaries.last().copied().flatten() != Some(Boundary::Clamp),
            "Tile::of: Boundary::Clamp cannot clamp the vectorized innermost axis (served at \
             {vector_size})"
        ));
        // Off the projection, not the space: a gathered operand's buffer has fewer physical axes
        // than its logical space has axes, and a storage-tiled one has more.
        let rank = comptime!(projection.physical_rank());
        let last = comptime!(rank - 1);
        let w = comptime!(vector_size as u32);
        let mut physical_shape = Coords::<u32>::new();
        let mut physical_strides = Coords::<u32>::new();
        #[unroll]
        for i in 0..rank {
            let extent = tensor.shape(i) as u32;
            let stride = tensor.stride(i) as u32;
            if comptime!(i == last) {
                // Innermost (contiguous, scalar stride 1): count lines; consecutive lines
                // are one line apart.
                physical_shape.push(extent / w);
                physical_strides.push(stride);
            } else {
                // Coarser axes re-express their scalar strides in lines.
                physical_shape.push(extent);
                physical_strides.push(stride / w);
            }
        }
        // Re-typing the buffer to the served scalar `T` is a static coercion for a plain
        // operand (the binding's real element is `Vector<T, w>`, same bytes); a quantized
        // store truly holds the stored type and the read view downcasts back (`lines_storage`).
        let buffer = unsafe {
            tensor
                .as_slice()
                .downcast_unchecked::<T>()
                .as_boxed_unchecked()
        };
        // `GmemLayout`'s own physical-position map: the operand's own projection relabeled by
        // position, since the layout is handed coordinates a gather has already resolved. Storage
        // tiling survives that relabeling, so `physical_shape` is exactly
        // `[pre…, grid…, …, tile…]` in synthetic-axis order.
        let gmem_projection = comptime!(projection.positional());
        // Logical bound folded from the physical shape, so it's correct for tiled
        // operands too (the physical buffer is padded; the logical extent is not).
        let bound = logical_extent(comptime!(gmem_projection.clone()), &physical_shape);
        // The whole-tile window. A `Dynamic` axis takes its runtime size from `bound`, so the
        // top-level extent never bakes into the kernel; a `Static` axis keeps its comptime size.
        let (origin, extent, map) = top_window(
            comptime!(space.clone()),
            &bound,
            &offsets,
            coefficients,
            vector_size,
            comptime!(coords.clone()),
        );
        Tile::<T> {
            tile_kind: TileKind::new_Gmem(MemData::<T> {
                store: Store::<T> {
                    buffer,
                    vector_size: comptime!(vector_size),
                    quant,
                    packing: comptime!(packing),
                },
                layout: GmemLayout {
                    physical_shape,
                    physical_strides,
                    projection: gmem_projection,
                },
                window: Window::new(
                    origin,
                    extent,
                    bound,
                    comptime!(coords.may_underflow()),
                    comptime!(spec.boundaries.clone()),
                ),
                source_window: ComptimeOption::new_None(),
                projection: comptime!(coords),
                map,
                offsets,
                window_start: 0u32,
                access: comptime!(Access {
                    whole: true,
                    overhang: if spec.is_checked() {
                        Overhang::Masked
                    } else {
                        Overhang::Fits
                    },
                    stage,
                }),
                lane_share: comptime!(LaneShare::Whole),
                init_from: comptime!(InitFrom::Cell),
            }),
            space: comptime!(space),
        }
    }
}

/// Comptime metadata bundled when constructing a shared-memory stage.
#[derive(Clone)]
pub(crate) struct StageMeta {
    pub space: Space,
    pub vector_size: usize,
    pub stage: StagePlan,
}

#[cube]
impl<T: Numeric> MemData<T> {
    /// Cooperatively materialize a coordinate-backed source into this plain, direct scalar memory
    /// tile. The caller must ensure that every unit in the cube owns this destination window:
    /// workers write cyclic positions across it. That is a property of the level's distribution,
    /// rather than whether this window covers its backing buffer.
    pub(crate) fn fill_procedural(&mut self, src: &ProceduralData<T>, #[comptime] space: Space) {
        comptime!(assert!(
            self.store.packing == Packing::Plain
                && self.projection.is_direct()
                && self.store.vector_size == 1,
            "MemData::fill_procedural: procedural sources require a plain, direct scalar destination"
        ));
        // Read the destination's runtime window rather than the comptime space so direct copies
        // also work when another operand witnesses a Dynamic extent.
        let shape = self.window.extent.clone();
        let mut dst = self.flat_mut::<Const<1>>();
        let total = dst.shape();
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            let pos = unravel(&shape, i.fcast::<u32>());
            // TODO: Staging has no knowledge of its eventual consumer, so this generic masked
            // fill uses ProceduralData's zero fallback. That is not the identity for every
            // reduction (notably Max over negative values and Min over positive values). A
            // reduction-aware staging contract must carry either validity or the consumer's
            // identity into this fill before staged procedural reductions can be fully correct.
            dst.write(
                i,
                Vector::cast_from(src.evaluate_masked(&pos, comptime!(space.clone()))),
            );
            i += workers;
        }
    }

    /// Allocate a fresh shared-memory tile shaped to stage one `divide()` sub-tile of `operand`, in
    /// the element that operand needs staged: the one it *serves* when the load is what decodes it
    /// ([`DequantAt::Load`], and always for a plain operand, whose served and stored elements are the
    /// same), else the one it is *stored* in ([`smem_stored`](MemData::smem_stored)). The operand
    /// carries which, so a caller stages it without asking whether it is quantized at all.
    pub fn smem_like(operand: &Tile<T>) -> Tile<T> {
        let dequant_at = operand.dequant_at();
        match comptime!(dequant_at) {
            DequantAt::Load => {
                let space = comptime!(operand.space.divide());
                let projection = operand.projection();
                // The stage is one level down, so it takes the operand's plan from the next level
                // on: its own residence was consumed by the decision to build it.
                let source_plan = operand.stage_plan();
                let stage = comptime!(source_plan.descend());
                // A padded stage is served wider than the operand it is filled from: the buffer
                // owns its own layout, so an axis global memory could not vectorize still reaches
                // the leaf in lines. `fill_straight` widens across the two.
                let source_width = operand.vector_size();
                let vector_size = comptime!(source_plan.effective_width(source_width));

                if comptime!(projection.is_direct()) {
                    MemData::smem(space, vector_size, stage)
                } else {
                    MemData::smem_gathered(
                        space,
                        vector_size,
                        stage,
                        projection,
                        &operand.runtime_map(),
                        operand.window_signed(),
                        operand.window_boundaries(),
                    )
                }
            }
            DequantAt::Read => MemData::smem_stored(operand),
        }
    }

    /// [`smem_like`](MemData::smem_like) in the element the operand is *stored* in rather than the
    /// one it serves: a quantized operand keeps its stored form (native `i8`, or `u32` words when
    /// the scheme packs several values each) and its scales, so the leaf dequantizes at the read
    /// (see [`smem_quant`](MemData::smem_quant)) instead of the fill inflating the stage to `T`.
    /// Reached only through [`smem_like`](MemData::smem_like), on an operand whose quantized form
    /// runs [`DequantAt::Read`].
    fn smem_stored(operand: &Tile<T>) -> Tile<T> {
        let space = comptime!(operand.space.divide());
        let vector_size = operand.vector_size();
        let source_plan = operand.stage_plan();
        let stage = comptime!(source_plan.descend());
        match &operand.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                #[comptime]
                match &g.store.quant {
                    // Served == stored, so this is `smem_like`.
                    ComptimeOption::None => MemData::smem(space, vector_size, stage),
                    ComptimeOption::Some(info) => match comptime!(info.scheme.store) {
                        QuantStore::Native => match comptime!(info.scheme.value) {
                            QuantValue::Q8F | QuantValue::Q8S => MemData::smem_quant::<i8>(
                                space,
                                vector_size,
                                stage,
                                info.table.clone(),
                                comptime!(info.scheme),
                            ),
                            other => panic!(
                                "MemData::smem_stored: native quant storage element {:?} is not wired (i8 only)",
                                other
                            ),
                        },
                        QuantStore::PackedU32(_) => MemData::smem_quant::<u32>(
                            space,
                            vector_size,
                            stage,
                            info.table.clone(),
                            comptime!(info.scheme),
                        ),
                        other => panic!(
                            "MemData::smem_stored: quant storage {:?} is not wired (native or packed-u32)",
                            other
                        ),
                    },
                }
            }
            // A tma source has no stored form to keep: it carries no scheme (`quantized` is a
            // strided-builder knob, and a tma tile is scalar), so served == stored. Giving it
            // one must not reuse this arm; see `Staging::new`, which refuses that combination.
            TileKind::TmaGmem(_) => MemData::smem(space, vector_size, stage),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("MemData::smem_stored: a fragment is not a stage source")
            }
            TileKind::Procedural(_) => {
                panic!("MemData::smem_stored: a procedural tile is not a stage source")
            }
        }
    }

    /// Allocate a shared-memory tile over `space`, at physical `vector_size` (the slice is
    /// allocated natively wide, then scalar-erased). A `Tiled` stage is storage-tiled at the
    /// final tile (one contiguous block per fragment, what a cmma transaction wants) so the cmma
    /// transaction reads it unstrided; `Strided` is plain row-major. A final-space stage has no
    /// grid to tile, so it is always plain. `units` is the launch's cube size, `0` when unknown.
    pub fn smem(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] stage: StagePlan,
    ) -> Tile<T> {
        let form = comptime!(StageForm::dense(&space, vector_size, stage.storage));
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        let meta = comptime!(StageMeta {
            space,
            vector_size,
            stage,
        });
        MemData::smem_with_form(meta, form, map, ComptimeOption::new_None())
    }

    /// [`smem`](MemData::smem) for a *gathered* operand: the stage holds the physical window its
    /// sub-tile reads, compacted ([`Compaction`]), rather than the logical tile. Several logical
    /// cells of a gather read the same physical one, so staging the logical tile would replicate
    /// elements by roughly the tap count; staging the window holds each one once.
    ///
    /// The stage therefore keeps the operand's own [`Projection`] (with the compaction's lattice
    /// quotiented out) instead of becoming direct: its buffer is the same shape of window the gmem
    /// operand was, so [`Tile::nd`] and [`at`](MemData::at) address it through exactly the machinery
    /// they address gmem through, and the fill stays a plain box copy.
    pub fn smem_gathered(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] stage: StagePlan,
        #[comptime] projection: Projection,
        map: &RuntimeMap,
        #[comptime] signed: bool,
        #[comptime] boundaries: SmallVec<[Option<Boundary>; MAX_AXES]>,
    ) -> Tile<T> {
        let form = comptime!(StageForm::gathered(
            &space,
            vector_size,
            stage.storage,
            &projection
        ));
        let stage_map = RuntimeMap {
            coefficients: map.coefficients.clone(),
            residues: const_coords(comptime!(vec![0; form.projection.physical_rank()])),
        };
        let stage_map =
            if comptime!(form.projection.is_rational() || form.projection.has_dynamic_scales()) {
                stage_map.stored()
            } else {
                stage_map
            };
        let meta = comptime!(StageMeta {
            space,
            vector_size,
            stage,
        });
        let source = MemData::<T>::pending_source_window(
            comptime!(form.steps.clone()),
            comptime!(signed),
            comptime!(boundaries),
        );
        MemData::smem_with_form(meta, form, stage_map, ComptimeOption::new_Some(source))
    }

    /// The body every smem constructor shares, taking the buffer's [`StageForm`] directly.
    fn smem_with_form(
        #[comptime] meta: StageMeta,
        #[comptime] form: StageForm,
        map: RuntimeMap,
        source: ComptimeOption<SourceWindow>,
    ) -> Tile<T> {
        let size!(W) = meta.vector_size;
        let smem = Shared::<[Vector<T, W>]>::new_slice(comptime!(form.cells()));
        MemData::smem_over(
            meta,
            &smem,
            ComptimeOption::new_None(),
            comptime!(Packing::Plain),
            form,
            map,
            source,
        )
    }

    /// [`smem`](MemData::smem) staging the element an operand is *stored* in rather than the one it
    /// serves: `I` is that element (`i8`, or `u32` when the scheme packs several values per word)
    /// and the line narrows to `vector_size / pack`, so the stage is that much smaller and a leaf
    /// dequantizes at the read instead of the fill inflating it to `T`. Carries a compact `Shared`
    /// scales buffer beside the values (one f32 per block of the sub-tile, refilled per region by
    /// [`fill_from`](MemData::fill_from)).
    pub fn smem_quant<I: Numeric>(
        #[comptime] space: Space,
        #[comptime] vector_size: usize,
        #[comptime] stage: StagePlan,
        table: ComptimeOption<Box<[f32]>>,
        #[comptime] scheme: QuantScheme,
    ) -> Tile<T> {
        // One stored line is one served line, just narrower, so only the element and width change:
        // the layout and window below are the same grid either way.
        let form = comptime!(StageForm::dense(&space, vector_size, stage.storage));
        let size!(WP) = comptime!(vector_size / scheme.num_quants());
        let smem = Shared::<[Vector<I, WP>]>::new_slice(comptime!(form.cells()));
        let quant = smem_quant_info(comptime!(space.clone()), table, comptime!(scheme));
        let map = RuntimeMap::integral(comptime!(form.projection.physical_rank()));
        let meta = comptime!(StageMeta {
            space,
            vector_size,
            stage,
        });
        MemData::smem_over(
            meta,
            &smem,
            quant,
            comptime!(scheme_packing(scheme)),
            form,
            map,
            ComptimeOption::new_None(),
        )
    }

    /// The body every smem constructor shares: everything but the allocation's element (which is why
    /// it takes the allocated slice rather than making it) and the buffer's [`StageForm`]. Scalar-
    /// erases the slice to the served `T` (the views recover the stored element through
    /// [`lines_storage`](MemData::lines_storage)) and wraps it in the whole-buffer window.
    fn smem_over<S: CubePrimitive>(
        #[comptime] meta: StageMeta,
        smem: &Shared<[S]>,
        quant: ComptimeOption<QuantInfo>,
        #[comptime] packing: Packing,
        #[comptime] form: StageForm,
        map: RuntimeMap,
        source: ComptimeOption<SourceWindow>,
    ) -> Tile<T> {
        let buffer = unsafe {
            smem.inner_ref()
                .downcast_unchecked::<T>()
                .as_boxed_unchecked()
        };
        let (physical_shape, physical_strides) = storage_layout(comptime!(form.clone()));
        let (origin, extent) = full_window(comptime!(form.clone()));
        // Smem never overhangs its own buffer, so the bound is the extent and checks are off.
        let bound = extent.clone();
        let gmem_projection = comptime!(form.positional.clone());
        Tile::<T> {
            tile_kind: TileKind::new_Smem(MemData::<T> {
                store: Store::<T> {
                    buffer,
                    vector_size: meta.vector_size,
                    quant,
                    packing: comptime!(packing),
                },
                layout: GmemLayout {
                    physical_shape,
                    physical_strides,
                    projection: gmem_projection,
                },
                // Stage origins are never negative, and smem never overhangs (`Overhang::Never`
                // below), so the boundary policy is never consulted. The empty list is the only
                // thing it *can* be: this window is shaped over the buffer's own dims (a tiled
                // stage's grid and tile fragments), while the sub-windows that inherit it are
                // shaped over coordinates, so any per-axis list minted here would land on the
                // wrong axes one level down.
                window: Window::new(origin, extent, bound, false, comptime!(SmallVec::new())),
                projection: comptime!(form.projection),
                map,
                offsets: Coords::<i32>::new(),
                window_start: 0u32,
                access: comptime!(Access {
                    whole: true,
                    overhang: Overhang::Never,
                    stage: meta.stage,
                }),
                lane_share: comptime!(LaneShare::Whole),
                init_from: comptime!(InitFrom::Cell),
                source_window: source,
            }),
            space: comptime!(meta.space),
        }
    }

    /// An unfilled [`SourceWindow`] for a gathered stage: the comptime geometry is the stage's own,
    /// while the origin and bound are written by each [`fill_from`](MemData::fill_from) from the
    /// operand that fill reads.
    fn pending_source_window(
        #[comptime] steps: SmallVec<[usize; MAX_AXES]>,
        #[comptime] signed: bool,
        #[comptime] boundaries: SmallVec<[Option<Boundary>; MAX_AXES]>,
    ) -> SourceWindow {
        let rank = comptime!(steps.len());
        let mut origin = Coords::<i32>::new();
        let mut bound = Coords::<u32>::new();
        #[unroll]
        for _ in 0..rank {
            origin.push(0i32);
            bound.push(0u32);
        }
        SourceWindow {
            origin,
            bound,
            steps: comptime!(steps),
            signed: comptime!(signed),
            boundaries: comptime!(boundaries),
        }
    }
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// A read [`View`] over `Vector<T, W>` lines: the scalar buffer re-grouped into its physical
    /// width, then re-viewed through the base layout and [`Window`]. `W` is the line width
    /// (`self.store.vector_size`); pass `Const<1>` when only the (width-invariant) leading shape is needed.
    pub fn view<W: Size>(&self) -> View<'_, Vector<T, W>, CoordsDyn> {
        match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                if comptime!(g.store.packing != Packing::Plain) {
                    panic!(
                        "Tile::view: a packed tile only serves values its read unpacks \
                         (Tile::copy_from, Tile::matrix_transparent)"
                    )
                }
                g.lines::<W>().view(g.base()).view(g.window())
            }
            TileKind::TmaGmem(_) => panic!("Tile::view: a tma source has no element view"),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::view: a plane tile has no memory view")
            }
            TileKind::Procedural(_) => panic!("Tile::view: a procedural tile has no memory view"),
        }
    }

    pub fn view_mut<W: Size>(&mut self) -> ViewMut<'_, Vector<T, W>, CoordsDyn> {
        match &mut self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => {
                if comptime!(g.store.quant.is_some()) {
                    panic!("Tile::view_mut: writing a quantized tile requires requantization")
                }
                let base = g.base();
                let window = g.window();
                g.lines_mut::<W>().view_mut(base).view_mut(window)
            }
            TileKind::TmaGmem(_) => panic!("Tile::view_mut: a tma source has no element view"),
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::view_mut: a plane tile has no memory view")
            }
            TileKind::Procedural(_) => panic!("Tile::view_mut: a procedural tile is not writable"),
        }
    }
}

#[cube]
impl<T: Numeric> MemData<T> {
    /// State what the accumulation being lowered starts from ([`init`](MemData::init)).
    pub(crate) fn set_init_from(&mut self, #[comptime] init_from: InitFrom) {
        comptime!({
            self.init_from = init_from;
        });
    }

    /// Memory transport leaf: cooperative cyclic copy of `src` into `self`, whole
    /// `Vector<T, W>` lines at `self`'s width, unit `u` moving lines `u`, `u + CUBE_DIM`, ….
    /// The caller owns the rendezvous: a `sync_cube` must separate this fill from its readers.
    ///
    /// `space` is the logical space both sides carry. A gathered `src` stages into the compacted
    /// copy of the *window* it reads rather than of its logical tile, so the fill stays a box copy
    /// and the gather stays where it already was, at the leaf's read; see
    /// [`fill_straight`](MemData::fill_straight) and [`Compaction`].
    pub(crate) fn fill_from(&mut self, src: &MemData<T>, #[comptime] space: Space) {
        let size!(W) = comptime!(self.store.vector_size);
        let gathered = comptime!(!src.projection.is_direct());
        // A gathered stage keeps where it read from: the copy below writes the boundary's value
        // for every tap outside `src`, and nothing in the staged window says which those were.
        #[comptime]
        match &mut self.source_window {
            ComptimeOption::Some(source) => {
                source.origin = src.window.origin.clone();
                source.bound = src.window.bound.clone();
            }
            ComptimeOption::None => {}
        }
        if comptime!(self.store.quant.is_some()) {
            // Unreachable in practice: `Tile::of_impl` already asserts `quant.is_none() ||
            // coords.is_direct()` at construction, so a gathered `src` never carries a quantized
            // form to begin with. Kept as defense in case that invariant ever loosens.
            comptime!(assert!(
                !gathered,
                "MemData::fill_from: a gathered operand cannot stage in its quantized form"
            ));
            // Quant → quant: stage the packed storage words verbatim through the straight-line fill,
            // then the scales beside them, so a leaf read dequantizes straight out of smem with no
            // f32 inflation. A quantized stage is always a fresh whole buffer, so the masked slow
            // path never applies.
            comptime!(assert!(
                self.access.whole && !self.access.overhang.masks(),
                "MemData::fill_from: a quantized stage is always a fresh whole buffer"
            ));
            #[comptime]
            match &src.store.quant {
                ComptimeOption::Some(info) => match comptime!(info.scheme.store) {
                    // Unpacked: one element per value, so the physical line is the served line.
                    QuantStore::Native => match comptime!(info.scheme.value) {
                        QuantValue::Q8F | QuantValue::Q8S => {
                            self.fill_straight::<i8, W>(src, comptime!(space.clone()))
                        }
                        other => panic!(
                            "MemData::fill_from: native quant storage element {:?} is not wired (i8 only)",
                            other
                        ),
                    },
                    // Packed: the buffer holds `u32`s carrying `num_quants` values each, so the
                    // physical line is that much narrower than the served one.
                    QuantStore::PackedU32(_) => {
                        let size!(WP) =
                            comptime!(self.store.vector_size / info.scheme.num_quants());
                        self.fill_straight::<u32, WP>(src, comptime!(space.clone()));
                    }
                    other => panic!(
                        "MemData::fill_from: quant storage {:?} is not wired (native or packed-u32)",
                        other
                    ),
                },
                ComptimeOption::None => panic!(
                    "MemData::fill_from: a quantized stage must be filled from a quantized source"
                ),
            }
            self.stage_scales(src);
        } else if comptime!(
            self.access.whole
                && !self.access.overhang.masks()
                && src.store.packing == Packing::Plain
        ) {
            // Plain → plain, whole destination: fill in destination-physical order (the write is
            // linear and only the source decodes, once per line by constants on a static store).
            // A padded stage is served in lines its source cannot hand out whole, so assemble each
            // destination line lane by lane.
            if comptime!(self.store.vector_size != src.store.vector_size) {
                comptime!(assert!(
                    src.store.vector_size == 1,
                    "MemData::fill_from: a padded stage is filled from a plain scalar operand, \
                     but its source serves {}-wide lines",
                    src.store.vector_size
                ));
            }
            self.fill_straight::<T, W>(src, comptime!(space.clone()));
        } else {
            // The general path reads the source as a flat run of its *window* and writes the
            // destination as a flat run of its own, which pairs the two only when they are the same
            // box. A gathered side's window is a physical box its logical rank does not match, so it
            // is addressed per axis or not at all. Reached by a windowed or masked destination, and
            // by a quantized source serving a plain one.
            comptime!(assert!(
                !gathered && self.projection.is_direct(),
                "MemData::fill_from: a gathered tile fills only a whole, unmasked, unquantized \
                 destination (a stage)"
            ));
            // The read decodes at the source's true storage element: `T` for a plain tile, else the
            // quantized store's element recovered from its scheme (the tile serves `T`, so `I` was
            // erased at construction and lives only on the scheme). This is what lets a plain
            // `copy_from`/`fill` dequantize on its own into a plain destination; the kernel never
            // threads `I`.
            #[comptime]
            match &src.store.quant {
                ComptimeOption::None => {
                    comptime!(assert!(
                        self.store.vector_size == src.store.vector_size,
                        "MemData::fill_from: a plain source is scanned at the destination's width, \
                         so a padded stage has to take the straight fill"
                    ));
                    // Equal widths here, so this only asks that the innermost extent is whole
                    // lines: `storage_extents` rounds it up, and nothing on the scan path would
                    // otherwise notice the last line the source cannot fill.
                    comptime!(fill_extent(
                        &space,
                        src.store.vector_size,
                        self.store.vector_size,
                        src.access.overhang.masks()
                    ));
                    // No scales to fold in, so the source's own packing is the whole read: served
                    // as stored, or unpacked from the words it holds into this plain destination.
                    let packing = src.packing();
                    match comptime!(packing) {
                        Packing::Plain => self.scan_transparent::<T, W, W>(src),
                        Packing::Native => panic!(
                            "MemData::fill_from: a native store with nothing to fold in serves \
                             its own element; bind it as that element"
                        ),
                        Packing::Packed { field: _ } => {
                            let size!(WP) = comptime!(packing.physical(src.store.vector_size));
                            self.scan_transparent::<u32, WP, W>(src)
                        }
                    }
                }
                ComptimeOption::Some(info) => match comptime!(info.scheme.store) {
                    QuantStore::Native => match comptime!(info.scheme.value) {
                        QuantValue::Q8F | QuantValue::Q8S => self.scan_transparent::<i8, W, W>(src),
                        other => panic!(
                            "MemData::fill_from: native quant storage element {:?} is not wired (i8 only)",
                            other
                        ),
                    },
                    QuantStore::PackedU32(_) => {
                        if comptime!(src.store.vector_size == self.store.vector_size) {
                            let size!(WP) =
                                comptime!(src.store.vector_size / info.scheme.num_quants());
                            self.scan_transparent::<u32, WP, W>(src)
                        } else {
                            // The source's line is one whole word and this stage
                            // is narrower: unpack each word across several lines.
                            self.scan_words::<W>(src)
                        }
                    }
                    other => panic!(
                        "MemData::fill_from: quant storage {:?} is not wired (native or packed-u32)",
                        other
                    ),
                },
            }
        }
    }

    /// The straight-line half of [`fill_from`](MemData::fill_from): a whole destination filled in
    /// destination-physical order, whole `Vector<I2, WP2>` lines (read whole for 1:1 copies, or
    /// assembled lane-by-lane from scalar sources for a padded stage), only the source decoding
    /// (once per line, by constants on a static store; half the address math of a logical-order scan).
    /// `I2` / `WP2` are the *storage* element and physical width: the served `(T, self.store.vector_size)`
    /// for a plain copy, the packed storage `(u32, served/pack)` (or native `i8`) for a quant stage.
    ///
    /// Both sides are physical boxes of the same rank here, whatever they are logically: the
    /// destination's coordinate is decoded once per line ([`physical_pos`]) and lands on the source
    /// cell it was staged from, so the fill copies and never gathers. A gathered pair differs only
    /// by the compaction's step ([`stage_compaction`]), which is `1` unless the source's window has holes
    /// to skip. The [`Window`] sits below either way, so a cell past the source's bound still masks
    /// to zero, and the stage holds that zero rather than re-masking at every read.
    fn fill_straight<I2: Numeric, WP2: Size>(
        &mut self,
        src: &MemData<T>,
        #[comptime] space: Space,
    ) {
        // A gathered stage owns mutable map registers alongside its bytes. Store the source
        // window's coefficients and phase into those registers so bytes and interpretation are
        // one slot value. Direct stages carry no runtime map state.
        if comptime!(self.projection.is_rational() || self.projection.has_dynamic_scales()) {
            self.map.store_from(&src.map);
        }
        let check = comptime!(src.access.overhang.masks());
        let sw = comptime!(src.store.vector_size);
        let w = comptime!(self.store.vector_size);
        let compaction = comptime!(stage_compaction(
            &src.projection,
            &self.projection,
            w,
            &space
        ));
        // Empty exactly when the window has no holes to skip, so the fill reads the source box
        // straight through and this layer is never built.
        let steps = comptime!(match &compaction {
            Some(c) if !c.is_dense() => c.steps().to_vec(),
            _ => Vec::new(),
        });
        let shape = self.layout.physical_shape.clone();
        let plen = shape.len().comptime();
        let total = shape
            .fproduct(comptime!((0..plen).collect::<Vec<_>>()))
            .fcast::<usize>();
        let projection = comptime!(self.layout.projection.clone());
        // Asked whatever the widths: an equal-width fill reads nothing off the extent, but owes
        // the same agreement between the two boxes.
        let lanes = comptime!(fill_extent(&space, sw, w, check));
        let src_rank = comptime!(src.projection.physical_rank());
        let padding = comptime!((sw != w).then(|| {
            // `source_lane` swaps the innermost entry of a destination coordinate to address the
            // source, which only lands on a source cell when the two boxes have the same rank. A
            // storage-tiled stage splits each axis into a grid and a block digit and does not.
            assert!(
                src_rank == plen,
                "MemData::fill_straight: a padded stage is a rank-{plen} box filled from a \
                 rank-{src_rank} source, so a destination coordinate does not address it"
            );
            Padding {
                width: w,
                lanes,
                rank: src_rank,
            }
        }));
        // A comptime worker count emits the tasks straight-line: a rolled loop's runtime `CUBE_DIM`
        // stride blocks unrolling, and on Metal's in-order pipe each line's store then stalls the
        // next line's read. Only a spilling last task needs its guard; unknown or tiny cubes take
        // the rolled loop. `constant()` bridges the folded total back to host data; a whole smem
        // stage's shape is static, so it always folds.
        let units = comptime!(self.access.stage.units);
        let total_c = total.constant();
        // The other half of the fill's contract: the mappings agree ([`stage_compaction`]), and so
        // do the sizes. A gathered destination is always an smem stage, so its line count folds and
        // has to be exactly the compacted window's.
        let cells = comptime!(compaction.as_ref().map(|c| c.cells(w)));
        comptime!(assert!(
            match cells {
                Some(n) => matches!(total_c, Some(t) if t as usize == n),
                None => true,
            },
            "MemData::fill_straight: a gathered source fills a destination of {total_c:?} lines, \
             but its compacted window is {cells:?}"
        ));
        let straight =
            comptime!(matches!(total_c, Some(t) if units > 0 && (t as usize).div_ceil(units) <= 8));
        let d = self.lines_storage_mut::<I2, WP2>();
        if comptime!(sw == w) {
            let s = if comptime!(steps.is_empty()) {
                MaskedView::new(
                    src.lines_storage::<I2, WP2>()
                        .view(src.base())
                        .view(src.window()),
                    check,
                )
            } else {
                MaskedView::new(
                    src.lines_storage::<I2, WP2>()
                        .view(src.base())
                        .view(src.window())
                        .view(StepUp::new(shape.clone(), comptime!(steps))),
                    check,
                )
            };
            fill_lines::<I2, WP2, WP2>(
                d, &s, projection, &shape, total, total_c, units, straight, padding,
            );
        } else {
            let s = if comptime!(steps.is_empty()) {
                MaskedView::new(
                    src.lines_storage::<I2, Const<1>>()
                        .view(src.base())
                        .view(src.window()),
                    check,
                )
            } else {
                MaskedView::new(
                    src.lines_storage::<I2, Const<1>>()
                        .view(src.base())
                        .view(src.window())
                        .view(StepUp::new(
                            widened_shape(&shape, comptime!(plen), comptime!(w)),
                            comptime!(steps),
                        )),
                    check,
                )
            };
            fill_lines::<I2, WP2, Const<1>>(
                d, &s, projection, &shape, total, total_c, units, straight, padding,
            );
        }
    }

    /// Refill this quantized stage's scales side-channel from `src` for the current region: one f32
    /// per block of the sub-tile, from `src`'s windowed scales into the compact self-relative grid
    /// [`smem_quant_info`] laid out. Cooperative across the cube (one block per task, cyclic). The
    /// destination index is the flat block index itself (the grid is row-major, so the per-axis
    /// decode inverts exactly); the source index dots the block coords with `src`'s scale strides,
    /// whose `window_start` already carries the region's base block.
    fn stage_scales(&mut self, src: &MemData<T>) {
        let dst = self.store.quant.as_mut().unwrap();
        let sinfo = src.store.quant.as_ref().unwrap();
        let nb = comptime!(dst.scale_shape.clone());
        let rank = comptime!(nb.len());
        let count = comptime!(nb.iter().product::<usize>());
        let dend = dst.buffer.len();
        let dst_scales = dst.buffer.slice_mut(0, dend);
        let send = sinfo.buffer.len();
        let src_scales = sinfo.buffer.slice(0, send);
        let workers = CUBE_DIM as usize;
        let mut bl = UNIT_POS as usize;
        while bl < count {
            let x = bl.fcast::<u32>();
            let mut src_idx = sinfo.window_start;
            #[unroll]
            for p in 0..rank {
                let after = comptime!(nb[(p + 1)..].iter().product::<usize>());
                let bi = x
                    .fdiv(comptime!(after as u32))
                    .frem(comptime!(nb[p] as u32));
                src_idx = src_idx.fadd(bi.fmul(sinfo.strides.at(p)));
            }
            // The grid holds *effective* scales: a two-level source's global level folds in here,
            // once per block per stage, so everything below the stage serves a one-level scheme
            // and no global scale threads past this point.
            dst_scales[bl] = sinfo.known.effective(src_scales[src_idx.fcast::<usize>()]);
            bl += workers;
        }
    }

    /// Zero this window: whole lines at the store's width; a checked window skips
    /// cells past the logical bound.
    pub(crate) fn zero(&mut self) {
        self.init(T::from_int(0));
    }

    /// Initialize this window with `val`: whole lines at the store's width; a checked window
    /// skips cells past the logical bound.
    pub(crate) fn init(&mut self, val: T) {
        let size!(W) = comptime!(self.store.vector_size);
        let mut d = self.flat_mut::<W>();
        let total = d.shape();
        for i in 0..total {
            d.write(i, Vector::<T, W>::cast_from(val));
        }
    }

    /// The cooperative flat scan behind [`fill_from`](MemData::fill_from)'s general path: cyclic
    /// across the cube, each unit writing lines `u`, `u + CUBE_DIM`, …. Reads through
    /// [`flat_transparent`](MemData::flat_transparent) at storage element `I`, so a quantized
    /// source dequantizes into `T` transparently (`I == T` on a plain source).
    fn scan_transparent<I: Numeric, WP: Size, W: Size>(&mut self, src: &MemData<T>) {
        let s = src.flat_transparent::<I, WP, W>();
        let mut d = self.flat_mut::<W>();
        let total = d.shape();
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            // `src` zeroes reads past its logical bound (the partial-tile overhang); the
            // staged buffer is unchecked, so the full padded cell is still written.
            d.write(i, s.read(i));
            i += workers;
        }
    }

    /// The sub-word twin of [`scan_transparent`](MemData::scan_transparent): the source's served
    /// line is one whole packed word (`vector_size == num_quants`, a scalar `u32` binding), and
    /// each word unpacks into `num_quants / W` lines of this store's width: how a packed operand
    /// fills a stage on a device whose vectors cannot cover a word. Word-serving is what keeps the
    /// line/storage-line correspondence exact (one line **is** one word), so no other width plays.
    ///
    /// Unchecked only, and unreachable any other way: a checked operand cannot vectorize
    /// ([`realize`](crate::StridedTileSource) refuses it), and a word-serving operand is
    /// `num_quants` wide, so a checked source never gets here; the assert below is a backstop for
    /// hand-built args. The ragged-tail obligation this leaves is the engine's ordinary unchecked
    /// contract, stated at the operand: a cut that overhangs the buffer *panics at launch* unless
    /// the caller declared `checked(false)`, and that declaration is the caller's claim that every
    /// staged block lies inside the allocation (e.g. an S block pinned to a divisor of the cache's
    /// capacity) with consumption clipped at the leaves. The innermost scale block must cover
    /// whole words, so a word never straddles two scales.
    fn scan_words<W: Size>(&mut self, src: &MemData<T>) {
        #[comptime]
        match &src.store.quant {
            ComptimeOption::Some(info) => {
                let nq = comptime!(info.scheme.num_quants());
                comptime!(assert!(
                    src.store.vector_size == nq,
                    "MemData::scan_words: the source serves whole words (vector_size == num_quants)"
                ));
                let w = comptime!(self.store.vector_size);
                comptime!(assert!(
                    w < nq && nq.is_multiple_of(w),
                    "MemData::scan_words: the stage width must divide the packing factor"
                ));
                comptime!(assert!(
                    !src.access.overhang.masks(),
                    "MemData::scan_words: a sub-word fill reads unchecked"
                ));
                comptime!(assert!(
                    info.block.last().unwrap().is_multiple_of(nq),
                    "MemData::scan_words: the innermost scale block must cover whole words"
                ));
                let lpw = comptime!(nq / w);
                let size!(NW) = 1usize;
                let words = src
                    .lines_storage::<u32, NW>()
                    .view(src.base())
                    .view(src.window())
                    .view(FlatLayout::new(src.window.extent.clone()));
                let scales = info
                    .buffer
                    .view(ScaleLayout::new(
                        info.strides.clone(),
                        info.window_start,
                        comptime!(info.block.clone()),
                        comptime!(src.store.vector_size),
                        comptime!(info.extent.clone()),
                    ))
                    .view(FlatLayout::new(src.window.extent.clone()));
                let mut d = self.flat_mut::<W>();
                let total = d.shape();
                let workers = CUBE_DIM as usize;
                let mut i = UNIT_POS as usize;
                while i < total {
                    let word = words.read(i / lpw).extract(0usize);
                    let scale = scales.read(i / lpw);
                    let first = ((i % lpw) * w) as u32;
                    let vals = unpack_fields::<T, W>(
                        word,
                        first,
                        info.table.clone(),
                        comptime!(info.scheme),
                    );
                    d.write(i, vals * Vector::new(T::cast_from(scale)));
                    i += workers;
                }
            }
            ComptimeOption::None => {
                panic!("MemData::scan_words: a plain source has no words to unpack")
            }
        }
    }

    /// Where this operand lives at each level below, and how a materialized level lays its buffer
    /// out; carried from the operand's [`TileSpec`].
    pub(crate) fn stage_plan(&self) -> comptime_type!(StagePlan) {
        comptime!(self.access.stage.clone())
    }

    /// How far this store's quantized form travels ([`DequantAt`]). A plain store answers
    /// [`DequantAt::Load`]: served and stored are the same element, so nothing is left to decode.
    // The `let`-then-return is load-bearing, see [`quant_pack`](MemData::quant_pack).
    #[allow(clippy::let_and_return)]
    pub(crate) fn dequant_at(&self) -> comptime_type!(DequantAt) {
        let dequant_at = #[comptime]
        match &self.store.quant {
            ComptimeOption::Some(info) => comptime!(info.dequant_at),
            ComptimeOption::None => DequantAt::Load,
        };
        dequant_at
    }

    /// How this store's values sit in memory, as stated at construction.
    pub(crate) fn packing(&self) -> comptime_type!(Packing) {
        comptime!(self.store.packing)
    }

    /// This buffer's byte length (its length is in native lines, so widened by the physical width):
    /// the transaction count a TMA fill into it lands. `T` / `vector_size` are served-typed, so a
    /// quantized buffer widens by the *storage* element and packed width instead (its line count is
    /// the same, one storage line per served line). Unreachable for quant today (only TMA smem
    /// destinations ask, and a TMA source never stages into a quantized register), but computed
    /// correctly rather than refused.
    pub(crate) fn size_bytes(&self) -> u32 {
        let lines = self.store.buffer.len() as u32;
        #[comptime]
        match &self.store.quant {
            ComptimeOption::None => {
                lines * T::size().comptime() as u32 * self.store.vector_size.comptime() as u32
            }
            ComptimeOption::Some(info) => {
                let wp = comptime!(self.store.vector_size / info.scheme.num_quants());
                match comptime!(info.scheme.store) {
                    QuantStore::Native => match comptime!(info.scheme.value) {
                        QuantValue::Q8F | QuantValue::Q8S => {
                            lines * i8::size().comptime() as u32 * wp as u32
                        }
                        other => panic!(
                            "MemData::size_bytes: native quant storage element {:?} is not wired (i8 only)",
                            other
                        ),
                    },
                    QuantStore::PackedU32(_) => lines * u32::size().comptime() as u32 * wp as u32,
                    other => panic!(
                        "MemData::size_bytes: quant storage {:?} is not wired (native or packed-u32)",
                        other
                    ),
                }
            }
        }
    }

    /// The base layout: the `[grid…, tile…]` split (`levels > 0`) or a plain
    /// strided dot (`levels = 0`).
    fn base(&self) -> GmemLayout {
        self.layout.clone()
    }

    fn window(&self) -> Window {
        self.window.clone()
    }

    /// The window extent, for shape-only readers that must not regroup the buffer.
    pub(crate) fn extent(&self) -> Coords<u32> {
        self.window.extent.clone()
    }

    /// The buffer re-grouped into `Vector<T, W>` lines, which the line-unit base/window
    /// layouts address. `W` is the width the buffer already has, so the regroup is a
    /// no-op; only the cmma row stride widens back to scalars ([`row_stride`](MemData::row_stride)).
    fn lines<W: Size>(&self) -> &[Vector<T, W>] {
        self.store.buffer.as_vectorized().with_vector_size::<W>()
    }

    /// The mutable twin of [`lines`](MemData::lines).
    fn lines_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        self.store
            .buffer
            .as_vectorized_mut()
            .with_vector_size_mut::<W>()
    }

    /// [`lines`](MemData::lines) with the buffer re-typed to the quantized storage
    /// element `I` it truly holds (see [`QuantInfo`]).
    fn lines_storage<I: Numeric, W: Size>(&self) -> &[Vector<I, W>] {
        let storage = unsafe { self.store.buffer.downcast_unchecked::<I>() };
        storage.as_vectorized().with_vector_size::<W>()
    }

    /// The mutable twin of [`lines_storage`](MemData::lines_storage): where a quant stage's
    /// [`fill_straight`](MemData::fill_straight) writes the packed storage words. `I == T` on a
    /// plain copy, a same-type reinterpret.
    fn lines_storage_mut<I: Numeric, W: Size>(&mut self) -> &mut [Vector<I, W>] {
        let storage = unsafe { self.store.buffer.downcast_mut_unchecked::<I>() };
        storage.as_vectorized_mut().with_vector_size_mut::<W>()
    }

    /// The window as one dense run of lines: index `i` addresses line
    /// `origin + i`: one add, no layout walk. Legal only where the window's
    /// content is physically contiguous in row-major order: an untiled,
    /// unmasked, unquantized store whose windowed logical axes are contiguous
    /// in memory (the strided operands a streaming fold windows). The
    /// comptime-checkable parts assert; the contiguity of the caller's
    /// layout is the caller's guarantee.
    pub(crate) fn dense_lines<W: Size>(&self) -> &[Vector<T, W>] {
        comptime!(assert!(
            !self.access.overhang.masks(),
            "MemData::dense_lines: a dense window cannot mask an overhang"
        ));
        comptime!(assert!(
            !self.layout.projection.is_tiled(),
            "MemData::dense_lines: a storage-tiled window is not dense"
        ));
        comptime!(assert!(
            self.projection.is_direct(),
            "MemData::dense_lines: a gathered window is not dense (sibling windows overlap)"
        ));
        if comptime!(self.store.packing != Packing::Plain) {
            panic!("MemData::dense_lines: a packed store is served through Tile::copy_from")
        }
        let all = self.lines::<W>();
        let start = self.window_start.fcast::<usize>();
        all.slice(start, all.len())
    }

    /// The mutable twin of [`dense_lines`](MemData::dense_lines).
    pub(crate) fn dense_lines_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        comptime!(assert!(
            !self.access.overhang.masks(),
            "MemData::dense_lines_mut: a dense window cannot mask an overhang"
        ));
        comptime!(assert!(
            !self.layout.projection.is_tiled(),
            "MemData::dense_lines_mut: a storage-tiled window is not dense"
        ));
        comptime!(assert!(
            self.projection.is_direct(),
            "MemData::dense_lines_mut: a gathered window is not dense (sibling windows overlap)"
        ));
        if comptime!(self.store.packing != Packing::Plain) {
            panic!("MemData::dense_lines_mut: a packed store cannot be written dense")
        }
        let start = self.window_start.fcast::<usize>();
        let all = self.lines_mut::<W>();
        let end = all.len();
        all.slice_mut(start, end)
    }

    /// The buffer from this window's origin on: the base a cmma load/store addresses,
    /// rows stepping by the scalar [`row_stride`](MemData::row_stride) (cmma takes a line
    /// slice with a scalar stride). Requires an unmasked store whose window doesn't split
    /// rows across storage tiles.
    pub(crate) fn window_slice(&self) -> &[T] {
        let offset = self.window_offset();
        self.store.buffer.slice(offset, self.store.buffer.len())
    }

    /// The mutable twin of [`window_slice`](MemData::window_slice).
    pub(crate) fn window_slice_mut(&mut self) -> &mut [T] {
        let offset = self.window_offset();
        let end = self.store.buffer.len();
        self.store.buffer.slice_mut(offset, end)
    }

    /// Line offset of the window origin: the accumulated `window_start`. On a tiled
    /// store the window must lie within one storage tile.
    fn window_offset(&self) -> usize {
        comptime!(assert!(
            !self.access.overhang.masks(),
            "MemData::window_offset: cmma cannot mask an overhang"
        ));
        // A raw window serves the buffer at the element it was erased to, so a quantized store
        // would hand its stored bytes over as served values. Every other door refuses the same way.
        if comptime!(self.store.packing != Packing::Plain) {
            panic!(
                "MemData::window_slice: a packed store has no raw element window; a fragment \
                 load reads it through Tile::matrix_transparent"
            )
        }
        self.window_start.fcast::<usize>()
    }

    /// Scalar stride between matrix rows: the line-unit physical stride of the leaf
    /// tile's row axis, widened back to scalars; a constant on a static store.
    pub(crate) fn row_stride(&self) -> u32 {
        let rank = comptime!(self.layout.projection.physical_rank());
        self.layout
            .physical_strides
            .at(comptime!(rank - 2))
            .fmul(comptime!(self.store.vector_size as u32).runtime())
    }

    /// Re-view this buffer through `layout` as a [`MaskedView`], carrying its own `check` flag
    /// so the leaf masks without being asked. `layout` is a [`TileMatrix`] for the 2-D matmul
    /// leaves and an [`AxisProjection`] for a gathered N-D read.
    pub(crate) fn masked<W: Size, C: Coordinates, L: TileLayout<C>>(
        &self,
        layout: L,
        #[comptime] guard: Guard,
    ) -> MaskedView<'_, Vector<T, W>, C> {
        if comptime!(self.store.packing != Packing::Plain) {
            panic!(
                "Tile::matrix: a packed tile only serves values its read unpacks \
                 (Tile::matrix_transparent)"
            )
        }
        MaskedView::new(
            self.lines::<W>()
                .view(self.base())
                .view(self.window().with_guard(guard))
                .view(layout),
            comptime!(guard.checks() && self.access.overhang.masks()),
        )
    }

    /// The mask flag a *write* view is built with: [`Overhang::masks`], plus the one policy a
    /// write cannot honour. [`Boundary::Clamp`] folds an out-of-range coordinate onto the edge
    /// cell instead of masking it, so several logical cells would write the same physical one;
    /// that is the aliasing [`matrix_mut`](MemData::matrix_mut) already refuses a gather for,
    /// arriving by a second route. Refused rather than silently raced.
    fn write_check(&self) -> comptime_type!(bool) {
        // Whole-operand on purpose, unlike the per-axis mask below it: one clamped axis is enough
        // to fold two distinct cells onto one, so there is no such thing as a partly writable
        // clamped operand.
        comptime!(assert!(
            !self.window.boundaries.contains(&Some(Boundary::Clamp))
                || !self.access.overhang.masks(),
            "MemData: a Boundary::Clamp operand is read-only, a clamped write aliases the edge cell"
        ));
        comptime!(self.access.overhang.masks())
    }

    /// The mutable twin of [`masked`](MemData::masked).
    pub(crate) fn masked_mut<W: Size, C: Coordinates, L: TileLayout<C>>(
        &mut self,
        layout: L,
    ) -> MaskedViewMut<'_, Vector<T, W>, C> {
        if comptime!(self.store.packing != Packing::Plain) {
            panic!("Tile::matrix_mut: writing a packed tile requires repacking")
        }
        let base = self.base();
        let window = self.window();
        let check = self.write_check();
        MaskedViewMut::new(
            self.lines_mut::<W>()
                .view_mut(base)
                .view_mut(window)
                .view_mut(layout),
            check,
        )
    }

    /// Re-view this buffer as a flat 1-D [`FlatView`] over its [`Window`] extent,
    /// carrying the `check` flag so a flat scan masks the overhang without being asked.
    pub(crate) fn flat<W: Size>(&self) -> FlatView<'_, Vector<T, W>> {
        FlatView::new(
            self.lines::<W>()
                .view(self.base())
                .view(self.window())
                .view(FlatLayout::new(self.window.extent.clone())),
            comptime!(self.access.overhang.masks()),
        )
    }

    /// Quantization-transparent [`flat`](MemData::flat): a plain store is read as it stands, a
    /// quantized one re-types to the storage element `I` and pairs it with the scales over the same
    /// window, dequantizing each read into `T`. `#[comptime]`, so the plain path pays nothing.
    pub(crate) fn flat_transparent<I: Numeric, WP: Size, W: Size>(
        &self,
    ) -> FlatView<'_, Vector<T, W>> {
        #[comptime]
        match &self.store.quant {
            ComptimeOption::Some(info) => {
                // The storage view groups at the *physical* width: a packed buffer holds
                // `W / num_quants` elements per served line.
                let values = self
                    .lines_storage::<I, WP>()
                    .view(self.base())
                    .view(self.window())
                    .view(FlatLayout::new(self.window.extent.clone()));
                let scales = info
                    .buffer
                    .view(ScaleLayout::new(
                        info.strides.clone(),
                        info.window_start,
                        comptime!(info.block.clone()),
                        comptime!(self.store.vector_size),
                        comptime!(info.extent.clone()),
                    ))
                    .view(FlatLayout::new(self.window.extent.clone()));
                let dequant = info.dequant_view::<I, WP, T, W, Coords1d>(values, scales);
                FlatView::new(dequant.view(), comptime!(self.access.overhang.masks()))
            }
            // The flat scan reads its own window whole, so it keeps the store's own mask.
            ComptimeOption::None => self.unscaled::<WP, W, Coords1d, FlatLayout>(
                FlatLayout::new(self.window.extent.clone()),
                comptime!(Guard::Checked),
            ),
        }
    }

    /// Quantization-transparent [`masked`](MemData::masked): the windowed twin of
    /// [`flat_transparent`](MemData::flat_transparent). A plain store is read as it stands; a
    /// quantized one re-types to the storage element `I`, pairs it with the scales over the same
    /// `layout`, and dequantizes each read into `T`. This is what lets a leaf read a
    /// quantized operand straight from gmem, or from a stage still in the stored element, without
    /// a dequantize-into-`f32` fill. `#[comptime]`, so plain pays nothing.
    pub(crate) fn transparent<
        I: Numeric,
        WP: Size,
        W: Size,
        C: Coordinates + 'static,
        L: TileLayout<C>,
    >(
        &self,
        layout: L,
        #[comptime] guard: Guard,
    ) -> MaskedView<'_, Vector<T, W>, C> {
        #[comptime]
        match &self.store.quant {
            // A quantized view *is* a view: cubecl's decodes on read and answers as `Vector<T, W>`,
            // so both arms hand back the same masked view and no caller learns the difference.
            ComptimeOption::Some(info) => {
                // The storage view groups at the *physical* width: a packed buffer holds
                // `W / num_quants` elements per served line.
                let values = self
                    .lines_storage::<I, WP>()
                    .view(self.base())
                    .view(self.window().with_guard(guard))
                    .view(layout.clone());
                // The scales over this same window: `ScaleLayout` resolves a window coordinate
                // to its block's scale, addressed by the same `layout` as the values, so both
                // answer the same coordinate.
                let scales = info
                    .buffer
                    .view(ScaleLayout::new(
                        info.strides.clone(),
                        info.window_start,
                        comptime!(info.block.clone()),
                        comptime!(self.store.vector_size),
                        comptime!(info.extent.clone()),
                    ))
                    .view(layout);
                let dequant = info.dequant_view::<I, WP, T, W, C>(values, scales);
                MaskedView::new(
                    dequant.view(),
                    comptime!(guard.checks() && self.access.overhang.masks()),
                )
            }
            ComptimeOption::None => self.unscaled::<WP, W, C, L>(layout, guard),
        }
    }

    /// The scale-free half of [`transparent`](MemData::transparent): a plain store read as it
    /// stands, a packed one unpacked at the read ([`PackedView`]). `WP` is the physical line the
    /// buffer holds ([`Packing::physical`]), `W` the served one.
    ///
    /// Where the quantized arm needs a scheme, a scale grid and a window start to say what a read
    /// means, this needs the field alone: a packed operand's values are values.
    fn unscaled<WP: Size, W: Size, C: Coordinates + 'static, L: TileLayout<C>>(
        &self,
        layout: L,
        #[comptime] guard: Guard,
    ) -> MaskedView<'_, Vector<T, W>, C> {
        let packing = self.packing();
        match comptime!(packing) {
            Packing::Plain => self.masked::<W, C, L>(layout, guard),
            Packing::Native => panic!(
                "MemData::transparent: a native store with nothing to fold in serves its own \
                 element; bind it as that element and let the contraction cast it"
            ),
            Packing::Packed { field } => {
                let words = self
                    .lines_storage::<u32, WP>()
                    .view(self.base())
                    .view(self.window())
                    .view(layout);
                let values = PackedView::<WP, T, W, C>::new(words, comptime!(field));
                MaskedView::new(
                    values.view(),
                    comptime!(guard.checks() && self.access.overhang.masks()),
                )
            }
        }
    }

    /// [`transparent`](MemData::transparent) over one batch matrix: what the 2-D matmul leaves
    /// read. `L` is [`TileMatrix`] for a direct operand and
    /// [`ProjectedMatrix`](super::ProjectedMatrix) for a gathered one; both answer the
    /// same [`Coords2d`] surface.
    pub(crate) fn matrix_transparent<I: Numeric, WP: Size, W: Size, L: TileLayout<Coords2d>>(
        &self,
        layout: L,
    ) -> MatrixView<'_, Vector<T, W>> {
        self.transparent::<I, WP, W, Coords2d, L>(layout, comptime!(Guard::Checked))
    }

    /// [`transparent`](MemData::transparent) over the tile's whole logical box, applying the
    /// operand's [`Projection`]: what a gather-reduce leaf reads, one coordinate per axis.
    pub(crate) fn nd_transparent<I: Numeric, WP: Size, W: Size>(
        &self,
        layout: AxisProjection,
        #[comptime] guard: Guard,
    ) -> MaskedView<'_, Vector<T, W>, CoordsDyn> {
        self.transparent::<I, WP, W, CoordsDyn, AxisProjection>(layout, guard)
    }

    /// [`nd_transparent`](MemData::nd_transparent) over the *physical* box instead of the logical
    /// one, for a caller that folds the map itself.
    ///
    /// The map is the only layer dropped: the [`Window`] that owns the boundary sits below it
    /// either way, so a coordinate past the operand's data masks exactly as it does through
    /// [`nd_transparent`](MemData::nd_transparent). The identity step keeps the box's own bound,
    /// which is the part a caller cannot fold away.
    ///
    /// The logical box test goes with the map, though, so this masks against the physical box
    /// alone. A caller owes it the coordinates the map would have produced from inside the logical
    /// one: a position folded from a logical coordinate out of range is no longer caught here, and
    /// reads whatever the window says lives at it.
    pub(crate) fn nd_physical<I: Numeric, WP: Size, W: Size>(
        &self,
    ) -> MaskedView<'_, Vector<T, W>, CoordsDyn> {
        let rank = comptime!(self.projection.physical_rank());
        let identity = StepUp::new(self.window.extent.clone(), comptime!(vec![1; rank]));
        // A folded caller hands in coordinates it derived itself, so it has proved nothing about
        // them: the window's boundary and the overhang mask both stay on.
        self.transparent::<I, WP, W, CoordsDyn, StepUp>(identity, comptime!(Guard::Checked))
    }

    /// The mutable twin of [`flat`](MemData::flat).
    pub(crate) fn flat_mut<W: Size>(&mut self) -> FlatViewMut<'_, Vector<T, W>> {
        if comptime!(self.store.packing != Packing::Plain) {
            panic!("Tile::flat_mut: writing a packed tile requires repacking")
        }
        let base = self.base();
        let window = self.window();
        let extent = self.window.extent.clone();
        let check = self.write_check();
        FlatViewMut::new(
            self.lines_mut::<W>()
                .view_mut(base)
                .view_mut(window)
                .view_mut(FlatLayout::new(extent)),
            check,
        )
    }

    /// The `i`-th batch matrix as a 2-D view. Mirrors [`Tile::matrix_mut`] for callers that
    /// hold the tile-kind rather than the whole tile, so the `space` is passed in.
    pub(crate) fn matrix_mut<W: Size>(
        &mut self,
        i: usize,
        #[comptime] space: Space,
    ) -> MatrixViewMut<'_, Vector<T, W>> {
        // The 2-D view reads its shape off the logical space, which is the physical shape only
        // under the direct mapping; a gathered operand is read through `Tile::nd`.
        comptime!(assert!(
            self.projection.is_direct(),
            "MemData::matrix_mut: a gathered operand has no plain 2-D matrix view"
        ));
        // Leading (batch) extents are width-invariant; the window extent is the view's shape.
        let bound = self.extent();
        let layout = batch_matrix(
            &bound,
            comptime!(&space),
            false,
            comptime!(self.store.vector_size),
            comptime!(MatrixAxes::trailing_pair(&space)),
            i,
        );
        self.masked_mut::<W, Coords2d, TileMatrix>(layout)
    }

    /// The [`AccumulateView`] over batch matrix `i`: [`matrix_mut`](MemData::matrix_mut) plus the
    /// [`LaneShare`] these cells carry, the [`Monoid`] they fold under and what the accumulation
    /// starts from, so a leaf accumulates through it without being told any of the three.
    pub(crate) fn matrix_accumulate<W: Size>(
        &mut self,
        i: usize,
        #[comptime] space: Space,
        #[comptime] monoid: Monoid,
    ) -> AccumulateView<'_, T, W> {
        let lane_share = comptime!(self.lane_share);
        let init_from = comptime!(self.init_from);
        AccumulateView::new(
            self.matrix_mut::<W>(i, space),
            lane_share,
            monoid,
            init_from,
        )
    }

    /// The [`AccumulateView`] over flat elements: [`flat_mut`](MemData::flat_mut) plus the
    /// [`LaneShare`] these cells carry and the [`Monoid`] they fold under.
    pub(crate) fn flat_accumulate<W: Size>(
        &mut self,
        #[comptime] monoid: Monoid,
    ) -> AccumulateView<'_, T, W, Coords1d> {
        // A flat logical scan only agrees with this physical window under the direct,
        // non-storage-tiled mapping. Otherwise the reduction's logical accumulator index would
        // seed and commit a different physical cell than the one it reduces for.
        comptime!(assert!(
            !self.layout.projection.is_tiled(),
            "MemData::flat_accumulate: a storage-tiled window has no flat logical accumulator view"
        ));
        comptime!(assert!(
            self.projection.is_direct(),
            "MemData::flat_accumulate: a gathered window has no flat logical accumulator view"
        ));
        let lane_share = comptime!(self.lane_share);
        let init_from = comptime!(self.init_from);
        AccumulateView::new(self.flat_mut::<W>(), lane_share, monoid, init_from)
    }

    /// Window down to `region`: shift the origin by the region's tile coordinate times the
    /// sub-tile edge, crop each physical axis to the region it now covers, re-box the same buffer.
    /// `bound` is carried through unchanged, so the leaf masks correctly at any nesting depth.
    ///
    /// Under a gathering [`Projection`] a physical axis is an affine combination of axes, so its
    /// advance sums one term per contributing axis and its extent is the receptive field
    /// ([`Projection::span`]) rather than a single edge: consecutive sibling windows overlap.
    pub(crate) fn at(&self, region: &Region, #[comptime] space: Space) -> MemData<T> {
        let mut origin = Coords::<i32>::new();
        let mut extent = Coords::<u32>::new();
        // Per-physical-axis window_start advances, summed below (chained, so constants fold).
        let mut advances = Coords::<u32>::new();

        let proj = comptime!(self.projection.clone());
        let rank = comptime!(proj.physical_rank());
        let last = comptime!(rank - 1);
        let w = comptime!(self.store.vector_size);

        let map = if comptime!(proj.is_direct()) {
            // One logical axis per physical axis at coefficient 1. Kept as its own loop because
            // this is the only mapping a *tiled* buffer can carry, where `step` folds the
            // grid/tile digit split that a scaled advance cannot be pushed through.
            #[unroll]
            for p in 0..rank {
                let axis = space.axis_at(p);
                // The innermost (vectorized) axis's edge is a line count, so `/ width`.
                let edge = comptime!(if p == last {
                    let e = space.partitioner().edge(axis);
                    // A padded stage's innermost extent need not fill whole lines, but then its
                    // partial tail line has no sibling to start after it: the axis has to be cut
                    // whole, or the next region would begin mid-line. `extent_raw` because a
                    // `Dynamic` axis has no extent to be cut whole, and owes the divisibility.
                    assert!(
                        e.is_multiple_of(w)
                            || matches!(space.extent_raw(axis), Extent::Static(x) if x == e),
                        "MemData::at: the innermost edge {e} is neither a whole number of \
                         {w}-wide lines nor the axis's whole extent ({:?}), so a region would \
                         start mid-line",
                        space.extent_raw(axis)
                    );
                    e.div_ceil(w)
                } else {
                    space.partitioner().edge(axis)
                });
                let index = region.coord(axis);

                origin.push(
                    self.window
                        .origin
                        .at(p)
                        .fadd(index.fmul(edge).fcast::<u32>().fcast::<i32>()),
                );
                extent.push(comptime!(edge as u32).runtime());
                advances.push(index.fcast::<u32>().fmul(step_offset(
                    comptime!(self.layout.projection.clone()),
                    comptime!(Axis(p as u8)),
                    edge,
                    &self.layout.physical_shape,
                    &self.layout.physical_strides,
                )));
            }
            // Every axis at coefficient 1, which no Dynamic term and no divisor can spell, so
            // there is nothing to carry and no phase to leave over.
            RuntimeMap::integral(rank)
        } else {
            let mut residues = Coords::<u32>::new();
            #[unroll]
            for pa in 0..rank {
                let (step, residue, span) = gathered_descent(
                    comptime!(proj.clone()),
                    comptime!(space.clone()),
                    region,
                    &self.map,
                    w,
                    pa,
                );

                // `step` only moves forward, so add directly to the signed origin.
                origin.push(self.window.origin.at(pa).fadd(step.fcast::<i32>()));
                residues.push(residue);
                extent.push(span);
                // `Projection::validate` pins a gathered operand to untiled storage (bare gmem, or
                // the row-major compacted stage of one), so one physical axis step is one stride
                // and the advance passes straight through.
                advances.push(step.fmul(self.layout.physical_strides.at(pa)));
            }
            // The coefficients are a fact about the buffer, invariant down the descent; only the
            // phase each axis's division left over is this level's.
            RuntimeMap {
                coefficients: self.map.coefficients.clone(),
                residues,
            }
        };
        let start = self
            .window_start
            .fadd(advances.fsum(comptime!((0..rank).collect::<Vec<_>>())));

        // Re-window the scales alongside the values.
        let mut origin_u32 = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            origin_u32.push(origin.at(p).fcast::<u32>());
        }
        let quant = #[comptime]
        match &self.store.quant {
            ComptimeOption::Some(info) => {
                comptime!(assert!(
                    !self.window.signed,
                    "MemData::at: a quantized operand cannot carry a negative window origin, its \
                     scale grid is addressed unsigned"
                ));
                // A quantized operand is direct (asserted at construction), so the child window's
                // extent per axis is this level's cut edge.
                ComptimeOption::new_Some(info.window(
                    &origin_u32,
                    rank,
                    comptime!(self.store.vector_size),
                    comptime!(
                        (0..rank)
                            .map(|p| space.partitioner().edge(space.axis_at(p)))
                            .collect::<Vec<_>>()
                    ),
                ))
            }
            ComptimeOption::None => ComptimeOption::new_None(),
        };

        MemData::<T> {
            store: Store::<T> {
                buffer: unsafe { self.store.buffer.as_boxed_unchecked() },
                vector_size: comptime!(self.store.vector_size),
                quant,
                packing: comptime!(self.store.packing),
            },
            // The layout addresses the whole buffer and never narrows; only the window moves.
            layout: self.layout.clone(),
            window: Window::new(
                origin,
                extent,
                self.window.bound.clone(),
                comptime!(self.window.signed),
                comptime!(self.window.boundaries.clone()),
            ),
            // How the logical axes address the physical ones is a fact about the buffer, invariant
            // down the descent. The offsets only placed the top window, which `origin` above
            // already carries.
            projection: comptime!(proj),
            // A region step moves this window and the source window by the same physical delta,
            // so the source window rides down as it was filled and only `origin` above moves.
            source_window: self.source_window.clone(),
            map,
            offsets: self.offsets.clone(),
            window_start: start,
            // The window no longer covers the buffer, so the straight-through fill is off. The
            // plan descends with the space: this level's residence is behind us now.
            access: comptime!(Access {
                whole: false,
                overhang: self.access.overhang,
                stage: self.access.stage.descend(),
            }),
            lane_share: comptime!(join_lane_share(self.lane_share, space.lane_share())),
            init_from: comptime!(self.init_from),
        }
    }
}

/// The whole-buffer window of a stage: `origin = 0`, `extent =` its own physical extents.
#[cube]
fn full_window(#[comptime] form: StageForm) -> (Coords<i32>, Coords<u32>) {
    let mut origin = Coords::<i32>::new();
    let mut extent = Coords::<u32>::new();

    #[unroll]
    for p in 0..comptime!(form.extents.len()) {
        origin.push(0);
        extent.push(comptime!(form.extents[p] as u32).runtime());
    }

    (origin, extent)
}

/// [`full_window`] for the top gmem tile, over the *physical* axes, where an axis may be
/// [`Dynamic`](crate::Extent): such an axis reads its runtime size from `bound` (the folded
/// logical extent) instead of a comptime constant, so the problem shape never specializes the
/// kernel. A gathered operand always reads `bound`: its physical axes are combinations of several
/// logical ones, so no single extent sizes one, and the whole buffer is the top window by
/// definition.
#[cube]
fn top_window(
    #[comptime] space: Space,
    bound: &Coords<u32>,
    offsets: &Coords<i32>,
    coefficients: Coords<u32>,
    #[comptime] vector_size: usize,
    #[comptime] projection: Projection,
) -> (Coords<i32>, Coords<u32>, RuntimeMap) {
    let mut origin = Coords::<i32>::new();
    let mut extent = Coords::<u32>::new();
    let mut residues = Coords::<u32>::new();
    let rank = comptime!(projection.physical_rank());
    let last = comptime!(rank - 1);

    #[unroll]
    for pa in 0..rank {
        let size = if comptime!(projection.is_direct()) {
            origin.push(0);
            residues.push(0u32);
            let axis = comptime!(space.axis_at(pa));
            // The innermost (vectorized) axis is a line count, `/ vector_size`. A `Dynamic` axis
            // reads its size from `bound`, already lined from the physical shape.
            match comptime!(space.extent_raw(axis)) {
                Extent::Static(e) => {
                    (comptime!(if pa == last { e / vector_size } else { e }) as u32).runtime()
                }
                Extent::Dynamic => bound.at(pa),
            }
        } else {
            let (start, phase) =
                gathered_origin(comptime!(projection.clone()), offsets, &coefficients, pa);
            origin.push(start);
            residues.push(phase);
            bound.at(pa)
        };
        extent.push(size);
    }

    (
        origin,
        extent,
        RuntimeMap {
            coefficients,
            residues,
        },
    )
}

/// Where a gathered physical axis's top window starts, and the phase its division left behind:
/// `⌊offset / divisor⌋` and `offset mod divisor`. An integer mapping divides by `1`, so its origin
/// absorbs the offset whole and its phase is `0`; a rational one can only absorb the multiples of
/// its divisor, and hands the rest to [`AxisProjection`](crate::AxisProjection), which adds it back
/// inside the numerator.
///
/// The floor is the host's whenever both sides are comptime
/// ([`PhysicalAxisMap::origin`](crate::PhysicalAxisMap::origin)); only a `Dynamic` offset or
/// divisor pays for one in the kernel.
#[cube]
fn gathered_origin(
    #[comptime] projection: Projection,
    offsets: &Coords<i32>,
    coefficients: &Coords<u32>,
    #[comptime] pa: usize,
) -> (i32, u32) {
    let axis_map = comptime!(projection.physical_axis(pa));

    if comptime!(axis_map.origin().is_some()) {
        (
            comptime!(axis_map.origin().unwrap() as i32).runtime(),
            comptime!(axis_map.residue().unwrap() as u32).runtime(),
        )
    } else {
        // A signed offset places the window before the buffer's origin (a padding), which is
        // exactly where truncating division would land a cell too high.
        let offset = match comptime!(axis_map.offset()) {
            Offset::Static(o) => comptime!(o as i32).runtime(),
            Offset::Dynamic => offsets.at(comptime!(projection.dynamic_offset_index(pa).unwrap())),
        };
        if comptime!(!axis_map.is_rational()) {
            (offset, 0u32)
        } else {
            let divisor = match comptime!(axis_map.divisor()) {
                Divisor::Static(d) => comptime!(d as i32).runtime(),
                Divisor::Dynamic { .. } => coefficients
                    .at(comptime!(projection.dynamic_divisor_index(pa).unwrap()))
                    .fcast::<i32>(),
            };
            let (start, residue) = floor_div_rem(offset, divisor);
            (start, residue.fcast::<u32>())
        }
    }
}

/// One gathered physical axis's descent into `region`: how far its window moves, the phase that
/// move leaves behind, and the receptive field the child then covers.
///
/// The move sums one term per contributing axis, its tile coordinate times its sub-tile edge times
/// its coefficient. All comptime but the coordinate, so this stays the multiply-add `window_start`
/// is documented to be. Each term divides by `vector_size` on its own, which only sums back to the
/// whole move because the innermost physical axis carries a single identity term:
/// [`Projection::validate`] requires it, precisely because that axis is addressed in lines. A
/// second term here would need the division after the sum, not before.
///
/// A [rational](crate::Divisor) axis moves by the whole cells its numerator crossed and hands the
/// phase it did not fill to the child: `⌊(move + phase)/d⌋` splits into this step plus a child
/// floor starting at the new phase, which is what makes the descent compose across levels.
#[cube]
fn gathered_descent(
    #[comptime] projection: Projection,
    #[comptime] space: Space,
    region: &Region,
    map: &RuntimeMap,
    #[comptime] vector_size: usize,
    #[comptime] pa: usize,
) -> (u32, u32, u32) {
    let axis_map = comptime!(projection.physical_axis(pa));
    let n = comptime!(axis_map.terms().len());
    let picks = comptime!((0..n).collect::<Vec<_>>());
    let lined = comptime!(pa == projection.physical_rank() - 1);

    let mut terms = Coords::<u32>::new();
    // One receptive-field term per contributing axis, `(edge - 1) * scale`. The field's leading
    // `1` is the branch's to add: under a division it is the quotient that carries it, not the
    // numerator.
    let mut spans = Coords::<u32>::new();
    #[unroll]
    for t in 0..n {
        let term = comptime!(axis_map.terms()[t]);
        let edge = comptime!(space.partitioner().edge(term.axis));
        match comptime!(term.scale) {
            Scale::Static(s) => {
                let step = comptime!(if lined {
                    edge * s / vector_size
                } else {
                    edge * s
                });
                terms.push(region.coord(term.axis).fmul(step).fcast::<u32>());
                spans.push(comptime!(((edge - 1) * s) as u32).runtime());
            }
            // The line division above never meets a runtime coefficient: the innermost physical
            // axis is a single identity term, which `Projection::validate` requires and `Static`
            // is the only spelling of.
            Scale::Dynamic { .. } => {
                let coefficient = map
                    .coefficients
                    .at(comptime!(projection.dynamic_scale_index(pa, t).unwrap()));
                terms.push(
                    region
                        .coord(term.axis)
                        .fcast::<u32>()
                        .fmul(comptime!(edge as u32).runtime())
                        .fmul(coefficient),
                );
                spans.push(comptime!((edge - 1) as u32).runtime().fmul(coefficient));
            }
        }
    }
    let advance = terms.fsum(comptime!(picks.clone()));

    if comptime!(!axis_map.is_rational()) {
        // The receptive field of the child edges: `1 + Σ (edge - 1) * scale`, which stays comptime
        // for the mapping that is.
        let span = if comptime!(!axis_map.has_dynamic_scale()) {
            comptime!({
                let s = projection.span(pa, |a| space.partitioner().edge(a));
                (if lined { s / vector_size } else { s }) as u32
            })
            .runtime()
        } else {
            spans.fsum(comptime!(picks.clone())).fadd(1)
        };
        (advance, 0u32, span)
    } else {
        // No `/ vector_size` anywhere below, and none is owed: `Projection::validate` refuses a
        // rational innermost physical axis at any width past `1`, so it is `1` whenever this
        // branch runs and the terms above are already in elements.
        let numerator = advance.fadd(map.residues.at(pa));
        let field = spans.fsum(comptime!(picks.clone()));
        match comptime!(axis_map.divisor()) {
            Divisor::Static(d) => {
                let d = comptime!(d as u32);
                let residue = numerator.frem(d);
                (
                    numerator.fdiv(d),
                    residue,
                    field.fadd(residue).fdiv(d).fadd(1),
                )
            }
            Divisor::Dynamic { .. } => {
                let d = map
                    .coefficients
                    .at(comptime!(projection.dynamic_divisor_index(pa).unwrap()));
                let residue = numerator.frem(d);
                (
                    numerator.fdiv(d),
                    residue,
                    field.fadd(residue).fdiv(d).fadd(1),
                )
            }
        }
    }
}

/// The staged scales side-channel for a quantized smem stage: a compact `Shared` buffer holding one
/// f32 per block of the sub-tile (row-major over the block grid), paired with self-relative strides
/// (`window_start = 0`). [`fill_from`](MemData::fill_from) refills its contents per region;
/// [`transparent`](MemData::transparent) reads it exactly as it reads a gmem operand's scales.
#[cube]
fn smem_quant_info(
    #[comptime] space: Space,
    table: ComptimeOption<Box<[f32]>>,
    #[comptime] scheme: QuantScheme,
) -> ComptimeOption<QuantInfo> {
    let rank = comptime!(space.rank());
    let block = comptime!(block_edges(scheme, rank));
    let (nb, strides_c) = comptime!(smem_scale_grid(&space, &block, scheme));
    let count = comptime!(nb.iter().product::<usize>());
    let scales = Shared::<[f32]>::new_slice(comptime!(count));
    let buffer = unsafe {
        scales
            .inner_ref()
            .downcast_unchecked::<f32>()
            .as_boxed_unchecked()
    };
    let mut strides = Coords::<u32>::new();
    #[allow(clippy::needless_range_loop)]
    #[unroll]
    for p in 0..rank {
        strides.push(comptime!(strides_c[p] as u32).runtime());
    }
    ComptimeOption::new_Some(QuantInfo {
        buffer,
        // The fill folds a two-level source's global level into the staged grid
        // ([`MemData::stage_scales`]), so the stage serves effective scales under the one-level
        // form of the scheme; keeping the two-level level here would fail cubecl's binding check.
        known: KnownScale::new_None(),
        strides,
        window_start: 0u32,
        block: comptime!(block),
        extent: comptime!(window_extents(&space, rank)),
        // A stage only keeps its quantized form when the read is what decodes it; that is the one
        // path reaching here.
        dequant_at: comptime!(DequantAt::Read),
        scale_shape: comptime!(nb),
        // The gmem table rides through: it is never staged, only re-read.
        table,
        scheme: comptime!(staged_scheme(scheme)),
    })
}

/// [`smem_quant_info`]'s host data: the per-axis distinct-scale count (`nb`) and its row-major
/// suffix-product strides. Per-tensor is the degenerate single scale; every count `1` and every
/// stride `0`, so a read pins index `0`; a block scheme grids `ceil(extent / block)` per axis.
fn smem_scale_grid(
    space: &Space,
    block: &[usize],
    scheme: QuantScheme,
) -> (Vec<usize>, Vec<usize>) {
    let rank = space.rank();
    let per_tensor = scheme.block_size().is_none();
    let nb: Vec<usize> = (0..rank)
        .map(|p| {
            if per_tensor {
                1
            } else {
                space.extent_at(p).div_ceil(block[p])
            }
        })
        .collect();
    let strides: Vec<usize> = (0..rank)
        .map(|p| {
            if per_tensor {
                0
            } else {
                nb[p + 1..].iter().product::<usize>()
            }
        })
        .collect();
    (nb, strides)
}

/// How a gathered fill's two sides relate: the [`Compaction`] of the source's own map, which the
/// destination must be addressed by, since that is what makes its physical box the window the
/// source was windowed down to. `None` when neither side gathers, which is every fill the engine
/// ran before compaction existed: there is nothing to compact, and no extent is read, since a
/// top-level `Dynamic` axis has none to read.
///
/// `space` is the destination's, used here to size the source's window. The assert pins the two
/// *mappings* together, which is what a caller not carrying the same axes on both sides
/// (`Tile::copy_from` is public) gets wrong; the two *sizes* are pinned separately, by
/// [`fill_straight`](MemData::fill_straight) against its own line count. `vector_size` is the
/// destination stage's served width, threaded to [`Compaction::of`].
fn stage_compaction(
    src: &Projection,
    dst: &Projection,
    vector_size: usize,
    space: &Space,
) -> Option<Compaction> {
    // A `MemData` carries the *coordinate*-space map ([`Projection::untiled`]), where storage
    // tiling has already folded back into the one coordinate its fragments are digits of. Direct
    // there is exactly "no gather", so a tiled buffer takes this early return like any other.
    if src.is_direct() && dst.is_direct() {
        return None;
    }
    let compaction = Compaction::of(src, vector_size, |axis| space.extent(axis));
    assert!(
        compaction.projection() == dst,
        "stage_compaction: a gathered source fills the compacted stage of its own \
         projection, addressed by {:?}, but the destination is addressed by {dst:?}",
        compaction.projection()
    );
    Some(compaction)
}

/// A stage's buffer: the physical extents it takes and the two mappings that address them. The one
/// place a dense stage and a gathered one differ, so [`smem_over`](MemData::smem_over) builds either
/// without knowing which it is.
#[derive(Clone, PartialEq, Eq, Debug)]
pub(crate) struct StageForm {
    /// Physical extents in lines, innermost already divided by the store width.
    extents: Vec<usize>,
    /// The buffer's own per-position map, what [`GmemLayout`] splits coordinates through.
    positional: Projection,
    /// How the staged tile's logical axes address those extents.
    projection: Projection,
    /// What a stage coordinate is multiplied by to land on the source, per physical axis. All `1`
    /// for a dense stage, which is a copy of the tile and shares its coordinates.
    steps: SmallVec<[usize; MAX_AXES]>,
}

impl StageForm {
    /// A materialized dense copy of the logical tile: what every direct operand stages into. An
    /// empty `nesting` is a plain row-major buffer; each block in it adds a `[grid…, block…]` split,
    /// so the buffer lays the innermost block down contiguously.
    fn dense(space: &Space, vector_size: usize, stage: StageStorage) -> StageForm {
        let nesting = stage_nesting(space, stage);
        StageForm {
            extents: storage_extents(space, vector_size, &nesting),
            positional: Projection::of_tiling(StorageTiling::uniform(space.rank(), nesting.len())),
            // A dense stage is a copy of the tile itself, so it addresses its own buffer directly
            // whatever the operand it stages was gathered through.
            projection: Projection::direct_over(space),
            steps: SmallVec::new(),
        }
    }

    /// The compacted window a gathered operand stages into ([`Compaction`]): one cell per element
    /// its sub-tile reads, addressed by the operand's own map with the lattice quotiented out.
    /// Always plain row-major, since an affine map cannot also be storage-tiled
    /// ([`Projection::validate`]).
    fn gathered(
        space: &Space,
        vector_size: usize,
        stage: StageStorage,
        projection: &Projection,
    ) -> StageForm {
        // `Tiled` comes from a cmma leaf, which `Staging::new` refuses a gathered operand for. The
        // nesting has nowhere to go here, so it is refused rather than silently dropped.
        assert!(
            matches!(stage, StageStorage::Strided),
            "StageForm: a gathered operand stages into a plain row-major window, but {stage:?} \
             storage was asked for"
        );
        let compaction = Compaction::of(projection, vector_size, |axis| space.extent(axis));
        let extents = compaction.line_extents(vector_size);
        StageForm {
            positional: Projection::of_tiling(StorageTiling::uniform(extents.len(), 0)),
            projection: compaction.projection().clone(),
            steps: compaction.steps().iter().copied().collect(),
            extents,
        }
    }

    /// How many lines the buffer holds.
    fn cells(&self) -> usize {
        self.extents.iter().product()
    }

    /// Row-major suffix-product strides over [`extents`](StageForm::extents).
    fn strides(&self) -> Vec<usize> {
        (0..self.extents.len())
            .map(|p| self.extents[p + 1..].iter().product())
            .collect()
    }
}

/// A stage's physical shape and strides, in lines like [`Tile::of`]'s.
#[cube]
fn storage_layout(#[comptime] form: StageForm) -> (Coords<u32>, Coords<u32>) {
    let strides_c = comptime!(form.strides());

    let mut shape = Coords::<u32>::new();
    let mut strides = Coords::<u32>::new();
    #[allow(clippy::needless_range_loop)]
    #[unroll]
    for p in 0..comptime!(form.extents.len()) {
        shape.push(comptime!(form.extents[p] as u32));
        strides.push(comptime!(strides_c[p] as u32));
    }

    (shape, strides)
}

/// The storage-tiling nesting a stage over `space` gets: the blocks its buffer lays down
/// contiguously, coarse to fine, each dividing the one before it (`space` is the implicit
/// outermost). Empty is a plain row-major buffer.
///
/// A `Tiled` stage groups the final tile, the block a cmma transaction reads unstrided. A final
/// space has no grid left to tile, so it stays plain whatever the layout asks for.
fn stage_nesting(space: &Space, stage: StageStorage) -> Vec<Space> {
    match stage {
        StageStorage::Tiled if !space.is_final() => vec![space.final_space()],
        _ => Vec::new(),
    }
}

/// A dense stage's physical line extents: `[extents…]` flat, or `[grid…, …, block…]`, one grid per
/// level of `nesting`. A level contributes how many of the next block down it holds; the innermost
/// contributes its own extents.
fn storage_extents(space: &Space, vector_size: usize, nesting: &[Space]) -> Vec<usize> {
    let rank = space.rank();
    let mut extents = Vec::new();
    let mut outer = space;
    for block in nesting {
        for p in 0..rank {
            let (e, b) = (outer.extent_at(p), block.extent_at(p));
            assert!(
                e.is_multiple_of(b),
                "MemData::smem: a {b}-element storage block must divide the {e}-element block \
                 enclosing it on axis {p}"
            );
            extents.push(e / b);
        }
        outer = block;
    }
    for p in 0..rank {
        extents.push(outer.extent_at(p));
    }
    // Rounded up, not truncated: a padded stage's innermost extent need not fill whole lines, and
    // the spare lanes of the last one are its padding. `fill_extent` refuses the case where the
    // rounding would instead mean the stage and its source disagree; every fill path asks it.
    let last = extents.len() - 1;
    extents[last] = extents[last].div_ceil(vector_size);
    extents
}

/// What a padded fill needs beyond the two boxes: `width` scalar source cells assembled per
/// destination line, and `lanes` the innermost extent past which those cells are padding. `None`
/// lanes is a `Dynamic` extent, where nothing is known at comptime and the source's own bounds
/// check is what zeroes them ([`fill_extent`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Padding {
    width: usize,
    lanes: Option<usize>,
    /// The physical rank both boxes share, which only this path needs: the 1:1 copy reads its
    /// line whole and never rebuilds a coordinate.
    rank: usize,
}

/// The innermost extent of `space` in cells, with the two widths a fill pairs checked against it.
///
/// The fill reads whole `sw`-wide source lines, so the innermost extent has to be a whole number
/// of them, and only the *destination* may hold a partial `w`-wide one. That partial line is what
/// a padded stage is, and its spare lanes hold zero. Without this the two boxes silently disagree,
/// the stage rounding its line count up ([`storage_extents`], `Compaction::line_extents`) where
/// the source truncated its own.
///
/// `None` for a `Dynamic` extent: nothing can be said at comptime, so a padded stage over one
/// leans on `check` to zero its spare lanes instead.
fn fill_extent(space: &Space, sw: usize, w: usize, check: bool) -> Option<usize> {
    match space.extent_raw(space.axis_at(space.rank() - 1)) {
        Extent::Static(e) => {
            assert!(
                e.is_multiple_of(sw),
                "MemData: the innermost extent {e} is not a whole number of the source's \
                 {sw}-wide lines, so the stage holds cells the source cannot hand it"
            );
            Some(e)
        }
        Extent::Dynamic => {
            assert!(
                sw == w || check,
                "MemData: a padded stage over a Dynamic innermost extent cannot know at comptime \
                 which lanes are padding, so its source must be bounds-checked for them to read \
                 as zero"
            );
            None
        }
    }
}

/// Schedule cooperative cyclic writing of destination stage lines across cube units.
///
/// Dispatches line reads via [`read_stage_line`], taking an unrolled loop when the task count
/// is small and static (`straight == true`) or a dynamic `CUBE_DIM`-strided while loop otherwise.
#[cube]
fn fill_lines<I2: Numeric, WP2: Size, SW: Size>(
    d: &mut [Vector<I2, WP2>],
    s: &MaskedView<'_, Vector<I2, SW>, CoordsDyn>,
    #[comptime] projection: Projection,
    shape: &Coords<u32>,
    total: usize,
    #[comptime] total_c: Option<u64>,
    #[comptime] units: usize,
    #[comptime] straight: bool,
    #[comptime] padding: Option<Padding>,
) {
    if comptime!(straight) {
        let tasks = comptime!((total_c.unwrap() as usize).div_ceil(units));
        #[unroll]
        for t in 0..tasks {
            let i = UNIT_POS as usize + comptime!(t * units);
            if comptime!((t + 1) * units > total_c.unwrap() as usize) {
                if i < total {
                    d[i] = read_stage_line::<I2, WP2, SW>(
                        s,
                        &physical_pos(comptime!(projection.clone()), i, shape),
                        comptime!(padding),
                    );
                }
            } else {
                d[i] = read_stage_line::<I2, WP2, SW>(
                    s,
                    &physical_pos(comptime!(projection.clone()), i, shape),
                    comptime!(padding),
                );
            }
        }
    } else {
        let workers = CUBE_DIM as usize;
        let mut i = UNIT_POS as usize;
        while i < total {
            d[i] = read_stage_line::<I2, WP2, SW>(
                s,
                &physical_pos(comptime!(projection.clone()), i, shape),
                comptime!(padding),
            );
            i += workers;
        }
    }
}

/// Read one destination line from the masked source view at `pos`: whole for a 1:1 copy, or
/// assembled lane by lane from scalar source cells for a padded stage ([`widen_line`]).
#[cube]
fn read_stage_line<I2: Numeric, WP2: Size, SW: Size>(
    s: &MaskedView<'_, Vector<I2, SW>, CoordsDyn>,
    pos: &CoordsDyn,
    #[comptime] padding: Option<Padding>,
) -> Vector<I2, WP2> {
    if comptime!(padding.is_some()) {
        widen_line::<I2, WP2, SW>(s, pos, comptime!(padding.unwrap()))
    } else {
        // The unpadded caller builds its view at the destination's own width, so `SW` *is* `WP2`
        // here and the cast is an identity the trace folds away; the two only differ as types.
        Vector::<I2, WP2>::cast_from(s.read(pos.clone()))
    }
}

/// The logical coordinate of physical line `i` in a `[grid…, tile…]` store: decode `i` into one
/// digit per physical axis ([`line_digit`]), then [`fold_physical`] folds a storage-tiled axis's
/// several digits back into one, off `projection`'s own div/modulo (`GmemLayout`'s synthetic
/// per-position map, invertible by construction).
#[cube]
fn physical_pos(#[comptime] projection: Projection, i: usize, shape: &Coords<u32>) -> CoordsDyn {
    let x = i.fcast::<u32>();
    let mut digits = Coords::<u32>::new();
    #[unroll]
    for j in 0..shape.len() {
        digits.push(line_digit(x, shape, j));
    }
    fold_physical(comptime!(projection), &digits, shape)
}

/// Assemble one padded destination line from adjacent scalar source cells.
///
/// When `Padding::lanes` is `None` (a `Dynamic` innermost extent), the source window must be
/// bounds-checked so that reads past the extent return zero. When it is `Some(n)`, reads past `n`
/// are masked off explicitly so the padding lanes keep the zero they start at.
#[cube]
fn widen_line<T: Numeric, W: Size, SW: Size>(
    s: &MaskedView<'_, Vector<T, SW>, CoordsDyn>,
    pos: &CoordsDyn,
    #[comptime] padding: Padding,
) -> Vector<T, W> {
    let width = comptime!(padding.width);
    let rank = comptime!(padding.rank);
    comptime!(assert!(
        SW::try_value_const() == Some(1),
        "widen_line: a padded stage is filled from a scalar source, got a {:?}-wide one",
        SW::try_value_const()
    ));
    comptime!(assert!(
        W::try_value_const().is_none_or(|n| n == width),
        "widen_line: assembles {width} lanes into a {:?}-wide destination line",
        W::try_value_const()
    ));
    let last = comptime!(rank - 1);
    let line = pos[last];
    let mut out = Vector::<T, W>::cast_from(T::from_int(0));
    let guarded = comptime!(match padding.lanes {
        Some(n) => !n.is_multiple_of(width),
        None => false,
    });
    #[unroll]
    for l in 0..width {
        let cell = line.fmul(comptime!(width as u32)).fadd(comptime!(l as u32));
        let valid = if comptime!(guarded) {
            cell < comptime!(padding.lanes.unwrap() as u32)
        } else {
            true.runtime()
        };
        if valid {
            out.insert(
                l,
                s.read(source_lane(pos, comptime!(rank), cell))
                    .extract(0usize),
            );
        }
    }
    out
}

/// Replace the destination line coordinate with its scalar source-cell coordinate.
#[cube]
fn source_lane(pos: &CoordsDyn, #[comptime] rank: usize, cell: u32) -> CoordsDyn {
    let mut out = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        if comptime!(p == rank - 1) {
            out.push(cell);
        } else {
            out.push(pos[p]);
        }
    }
    out
}

/// Express a padded destination's innermost physical extent in scalar source elements.
#[cube]
fn widened_shape(
    shape: &Coords<u32>,
    #[comptime] rank: usize,
    #[comptime] width: usize,
) -> Coords<u32> {
    let mut out = Coords::<u32>::new();
    #[unroll]
    for p in 0..rank {
        if comptime!(p == rank - 1) {
            out.push(shape.at(p).fmul(comptime!(width as u32)));
        } else {
            out.push(shape.at(p));
        }
    }
    out
}

/// Digit `j` of flat line `x` under `shape`'s row-major suffix strides.
#[cube]
fn line_digit(x: u32, shape: &Coords<u32>, #[comptime] j: usize) -> u32 {
    let plen = shape.len();
    x.fdiv(shape.fproduct(comptime!(((j + 1)..plen).collect::<Vec<_>>())))
        .frem(shape.at(j))
}

/// In-kernel twin of cubecl's `TiledViewLayout`, which has no in-kernel constructor: splits each
/// coordinate into the digits its [`projection`](Projection) spreads over the physical axes, then
/// dots the physical strides. Folding arithmetic, so a static store (smem) splits and dots by
/// constants, and an untiled projection (one physical axis per logical one, so every digit is the
/// whole coordinate) reduces to the plain strided dot. `Coordinates` are already physical (any
/// gather is resolved a layer up, by [`AxisProjection`]), so `projection` here is always
/// [`Projection::of_tiling`]'s synthetic per-position map, not the operand's own.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct GmemLayout {
    pub(crate) physical_shape: Coords<u32>,
    pub(crate) physical_strides: Coords<u32>,
    #[cube(comptime)]
    pub(crate) projection: Projection,
}

#[cube]
impl Layout for GmemLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        // Per-physical-axis terms, summed below (chained, so a static store's dot folds).
        let mut terms = Sequence::<u32>::new();
        let rank = comptime!(self.projection.physical_rank());
        #[unroll]
        for pa in 0..rank {
            let map = comptime!(self.projection.physical_axis(pa).clone());
            let picks = comptime!((0..map.terms().len()).collect::<Vec<_>>());
            // Almost always one term (only a gather layers several onto one physical axis, and
            // `GmemLayout`'s own map never does); summed the same way regardless.
            let mut parts = Sequence::<u32>::new();
            #[unroll]
            for t in 0..comptime!(map.terms().len()) {
                let term = comptime!(map.terms()[t]);
                let p = comptime!(self.projection.position(term.axis));
                let (finer, modulo) = comptime!(self.projection.digit(pa, term.axis));
                // Strip the finer digits, then take this one. The outermost fragment of an axis
                // (and any untiled axis) has no radix and keeps the full quotient.
                let quot = pos[p].fdiv(self.physical_shape.fproduct(comptime!(finer.to_vec())));
                let digit = match comptime!(modulo) {
                    Some(m) => quot.frem(self.physical_shape.at(m)),
                    None => quot,
                };
                parts.push(digit.fmul(comptime!(term.scale.get() as u32).runtime()));
            }
            terms.push(parts.fsum(picks).fmul(self.physical_strides.at(pa)));
        }
        terms
            .fsum(comptime!((0..rank).collect::<Vec<_>>()))
            .fcast::<usize>()
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        logical_extent(comptime!(self.projection.clone()), &self.physical_shape).to_dyn()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let bounds = self.shape();
        let mut valid = true;

        #[unroll]
        for i in 0..bounds.len() {
            valid = valid && pos[i] < bounds[i];
        }

        valid
    }
}

/// The layout [`Tile::at`] applies: shift every axis to `origin` and crop it to
/// `extent`. Same rank as the source; the rank-reducing 2-D slice is
/// [`TileMatrix`](super::TileMatrix).
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct Window {
    pub(crate) origin: Coords<i32>,
    pub(crate) extent: Coords<u32>,
    /// Absolute logical extent (the valid region). `shape()` stays `extent` (the tile
    /// cell, so loops cover the whole padded tile), but `is_in_bounds` clips against
    /// `bound` so a checked read/write zeroes / skips the overhang.
    pub(crate) bound: Coords<u32>,
    /// Whether the origin can be negative.
    #[cube(comptime)]
    pub(crate) signed: bool,
    /// Per-coordinate-axis boundary handling, same rank as `bound`. `None` means that axis is in
    /// bounds by construction; an empty list makes every axis `None`.
    #[cube(comptime)]
    pub(crate) boundaries: SmallVec<[Option<Boundary>; MAX_AXES]>,
}

#[cube]
impl Window {
    pub fn new(
        origin: Coords<i32>,
        extent: Coords<u32>,
        bound: Coords<u32>,
        #[comptime] signed: bool,
        #[comptime] boundaries: SmallVec<[Option<Boundary>; MAX_AXES]>,
    ) -> Self {
        // Both walks index `origin`, `pos` and `boundaries` by one counter, so a rank slip there
        // would silently apply one axis's mode to another rather than fail. `bound` is left out:
        // a sub-window inherits its parent's, which on a stage is the buffer's own rank (a tiled
        // stage's fragments) rather than the coordinate rank the origin is at.
        let origin_rank = origin.len();
        let extent_rank = extent.len();
        comptime!(assert!(
            extent_rank == origin_rank
                && (boundaries.is_empty() || boundaries.len() == origin_rank),
            "Window: origin ({origin_rank}), extent ({extent_rank}) and boundaries ({}) index the \
             same axes and must agree in rank",
            boundaries.len()
        ));
        Window {
            origin,
            extent,
            bound,
            signed,
            boundaries,
        }
    }

    /// Whether `pos` is valid on the selected physical axes. This is the factor-local form of
    /// [`Layout::is_in_bounds`]: separable normalization must mask the source axes moved by one
    /// tap without letting another factor's tap affect that decision.
    #[allow(clippy::needless_range_loop)] // `#[unroll]` requires a range loop.
    pub(crate) fn axes_in_bounds(&self, pos: &CoordsDyn, #[comptime] axes: Vec<usize>) -> bool {
        let mut valid = true;
        #[unroll]
        for a in 0..comptime!(axes.len()) {
            let i = comptime!(axes[a]);
            if comptime!(self.boundaries.get(i).copied().flatten() == Some(Boundary::Zero)) {
                valid = valid && self.axis_in_bounds(pos[i], i);
            }
        }
        valid
    }

    /// The scalar physical-axis check behind [`axes_in_bounds`](Self::axes_in_bounds).
    pub(crate) fn axis_in_bounds(&self, pos: u32, #[comptime] axis: usize) -> bool {
        if comptime!(self.boundaries.get(axis).copied().flatten() == Some(Boundary::Zero)) {
            let abs = self.origin.at(axis).fadd(pos.fcast::<i32>());
            if comptime!(self.signed) {
                abs >= 0i32 && abs.fcast::<u32>() < self.bound.at(axis)
            } else {
                abs.fcast::<u32>() < self.bound.at(axis)
            }
        } else {
            true.runtime()
        }
    }
}

/// Where a gathered stage sits inside the buffer it was filled from.
///
/// A stage is addressed by [`Compaction`](crate::Compaction)'s projection, which keeps the source
/// map's terms and drops its offset, so a staged coordinate `c` lands on `origin + c * step` in
/// the source. The fill wrote the boundary's value wherever that landed outside, and the staged
/// window cannot say which cells those were; this is what lets a reader put the question to the
/// source rectangle instead.
///
/// Invariant under [`at`](MemData::at): a region step moves the staged window and the source
/// window by the same physical delta, so only the staged origin has to move and this stays as it
/// was filled.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct SourceWindow {
    /// The source window's origin, as [`fill_from`](MemData::fill_from) found it.
    pub(crate) origin: Coords<i32>,
    /// The source buffer's logical extent, which is what a tap is in bounds against.
    pub(crate) bound: Coords<u32>,
    /// What a stage coordinate is multiplied by to land on the source, per physical axis.
    #[cube(comptime)]
    pub(crate) steps: SmallVec<[usize; MAX_AXES]>,
    /// Whether the source origin can be negative.
    #[cube(comptime)]
    pub(crate) signed: bool,
    /// The source's per-axis boundary handling. Same meaning as [`Window::boundaries`], read off
    /// the operand the stage was filled from rather than the stage's own (empty) list.
    #[cube(comptime)]
    pub(crate) boundaries: SmallVec<[Option<Boundary>; MAX_AXES]>,
}

#[cube]
impl SourceWindow {
    /// Whether one physical coordinate of the staged window lands inside the source.
    ///
    /// `stage_origin` is the staged window's own origin on this axis and `pos` the offset within
    /// it, so `stage_origin + pos` is the stage coordinate the caller is asking about. An axis the
    /// source did not pad is in bounds by construction, exactly as it is on a [`Window`].
    pub(crate) fn axis_in_bounds(
        &self,
        stage_origin: i32,
        pos: u32,
        #[comptime] axis: usize,
    ) -> bool {
        if comptime!(self.boundaries.get(axis).copied().flatten() == Some(Boundary::Zero)) {
            let step = comptime!(self.steps.get(axis).copied().unwrap_or(1) as i32);
            let cell = (stage_origin + pos.fcast::<i32>()) * step;
            let abs = self.origin.at(axis) + cell;
            if comptime!(self.signed) {
                abs >= 0i32 && abs.fcast::<u32>() < self.bound.at(axis)
            } else {
                abs.fcast::<u32>() < self.bound.at(axis)
            }
        } else {
            true.runtime()
        }
    }
}

#[cube]
impl Window {
    /// This window under `guard`: [`Guard::Proved`] drops the boundary machinery — no clamp for
    /// an origin that can go negative, and no per-axis [`Boundary`] mode — which is work whose
    /// answer the reader already knows and the window would otherwise pay for once per access.
    ///
    /// One path either way, differing only in comptime fields. A branch here would be a runtime
    /// select over the window's *runtime* halves as well, which is both slower and, on the
    /// accelerated leaves that share this constructor, wrong.
    pub(crate) fn with_guard(self, #[comptime] guard: Guard) -> Window {
        Window {
            origin: self.origin,
            extent: self.extent,
            bound: self.bound,
            signed: comptime!(guard.checks() && self.signed),
            boundaries: comptime!(match guard {
                Guard::Checked => self.boundaries.clone(),
                Guard::Proved => SmallVec::new(),
            }),
        }
    }
}

#[cube]
impl Layout for Window {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut out = CoordsDyn::new();

        #[unroll]
        for i in 0..self.origin.len() {
            let abs = self.origin.at(i).fadd(pos[i].fcast::<i32>());
            // Clamp negative coordinates to 0 before bounds masking. Branchless: this runs per
            // tap of every gathered read, where a diamond would cost more than the cast it skips.
            let shifted = if comptime!(self.signed) {
                select(abs >= 0i32, abs.fcast::<u32>(), 0u32)
            } else {
                abs.fcast::<u32>()
            };
            // Under `Clamp`, fold this coordinate onto its axis's edge cell rather than
            // leaving it for the mask.
            let shifted = match comptime!(self.boundaries.get(i).copied().flatten()) {
                Some(Boundary::Clamp) => {
                    let bound_i = self.bound.at(i);
                    let edge = select(shifted >= bound_i, bound_i.fsub(1u32), shifted);
                    // A zero-extent axis has no edge cell to fold onto, and the `bound - 1` above
                    // wrapped into a wild line index; both arms evaluate, so it is discarded here
                    // rather than skipped, and the axis folds to `0` like an underflow instead.
                    select(bound_i == 0u32, 0u32, edge)
                }
                None | Some(Boundary::Zero) => shifted,
            };
            out.push(shifted);
        }

        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.extent.to_dyn()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        self.axes_in_bounds(
            &pos,
            comptime!((0..self.boundaries.len()).collect::<Vec<_>>()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// `16 -> 8 -> 4` on both axes, so the space, its `divide()`, and its `final_space()` are
    /// three distinct block shapes to nest.
    fn space() -> Space {
        let seq = |edge| Cut::sequential(edge);
        Tiling::new()
            .extents(&[(M, 16), (N, 16)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(M, seq(8)).axis(N, seq(8))
            })
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(M, seq(4)).axis(N, seq(4))
            })
            .build()
    }

    /// [`space`] plus the ungathered innermost axis a gathered projection is required to carry.
    fn gathered_space() -> Space {
        let seq = |edge| Cut::sequential(edge);
        Tiling::new()
            .extents(&[(M, 16), (N, 16), (K, 8)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.axis(M, seq(8)).axis(N, seq(8)).axis(K, seq(4))
            })
            .build()
    }

    /// No nesting is the plain row-major buffer: the space's own extents, innermost in lines.
    #[test]
    fn flat_nesting_is_the_space_itself() {
        assert_eq!(storage_extents(&space(), 1, &[]), vec![16, 16]);
        assert_eq!(storage_extents(&space(), 4, &[]), vec![16, 4]);
    }

    /// One block is the `[grid…, tile…]` split: each axis holds `16 / 4` tiles of `4`.
    #[test]
    fn one_block_splits_grid_and_tile() {
        let space = space();
        assert_eq!(
            storage_extents(&space, 1, &[space.final_space()]),
            vec![4, 4, 4, 4]
        );
    }

    /// Two nested blocks add a middle grid, each level counting how many of the block below it
    /// it holds: `16 = 2 x (8 = 2 x (4))`.
    #[test]
    fn two_blocks_nest() {
        let space = space();
        let nesting = [space.divide(), space.final_space()];
        let extents = storage_extents(&space, 1, &nesting);
        assert_eq!(extents, vec![2, 2, 2, 2, 4, 4]);
        // The nesting only regroups the buffer, never resizes it.
        assert_eq!(extents.iter().product::<usize>(), space.tile_size());
    }

    /// The strides a buffer's extents imply, row-major: `[grid…, tile…]` or plain, the same rule.
    #[test]
    fn a_form_strides_row_major() {
        let space = space();
        let form = StageForm::dense(&space, 4, StageStorage::Strided);
        assert_eq!(form.extents, vec![16, 4]);
        assert_eq!(form.strides(), vec![4, 1]);
        assert_eq!(form.cells(), space.tile_size() / 4);

        let tiled = StageForm::dense(&space, 1, StageStorage::Tiled);
        assert_eq!(tiled.extents, vec![4, 4, 4, 4]);
        assert_eq!(tiled.strides(), vec![64, 16, 4, 1]);
    }

    /// A gathered stage is the compacted window, not the logical tile: `M` and `N` here map onto
    /// one physical axis, so the stage holds their receptive field instead of their product. `K`
    /// rides identity innermost, which every gathered projection is required to carry
    /// ([`Projection::validate`]).
    #[test]
    fn a_gathered_form_is_the_compacted_window() {
        let space = gathered_space().divide();
        let projection = Projection::new(
            &[M, N, K],
            &[
                PhysicalAxisMap::affine(&[(M, 1), (N, 1)]),
                PhysicalAxisMap::of(K),
            ],
        );
        let form = StageForm::gathered(&space, 1, StageStorage::Strided, &projection);
        // 8 x 8 logical cells over 1 + 7 + 7 physical ones, times the ungathered 4 of `K`.
        assert_eq!(form.extents, vec![15, 4]);
        assert_eq!(form.cells(), 60);
        assert_eq!(form.projection, projection);
        assert!(form.positional.is_direct());
    }

    /// A gathered operand's stage is plain row-major; a storage-tiled one has nowhere to nest.
    #[test]
    #[should_panic(expected = "plain row-major window")]
    fn a_gathered_form_refuses_tiled_storage() {
        let space = gathered_space().divide();
        let projection = Projection::new(
            &[M, N, K],
            &[
                PhysicalAxisMap::affine(&[(M, 1), (N, 1)]),
                PhysicalAxisMap::of(K),
            ],
        );
        StageForm::gathered(&space, 1, StageStorage::Tiled, &projection);
    }

    /// A block that does not divide the one enclosing it has no `[grid…, block…]` split.
    #[test]
    #[should_panic(expected = "must divide")]
    fn a_block_must_divide_its_enclosing_block() {
        let space = space();
        // Reversed: the coarse block sits inside the fine one.
        storage_extents(&space, 1, &[space.final_space(), space.divide()]);
    }

    /// A `Tiled` stage groups the final tile; a final space has no grid left, so it stays plain.
    #[test]
    fn stage_nesting_follows_the_layout() {
        let space = space();
        assert_eq!(
            stage_nesting(&space, StageStorage::Tiled),
            vec![space.final_space()]
        );
        assert!(stage_nesting(&space, StageStorage::Strided).is_empty());
        assert!(stage_nesting(&space.final_space(), StageStorage::Tiled).is_empty());
    }
}
