//! A launched memory operand ([`GlobalOperand`]): a tensor, a fused sink or a fused producer,
//! and the [`Memory`] tile it becomes, its top window boxed over the operand's physical axes.

use cubecl::{
    prelude::*,
    std::tensor::{ErasedTensor, WriteOnly},
};

use crate::*;

/// A launched memory operand, as the kernel receives it: a buffer, a fused sink or a fused
/// producer, with the geometry that addresses it and the statement the launch bound it under.
/// What a [`Memory`] tile is built from ([`tile`](Self::tile)); the kernel arguments build one.
#[derive(CubeType)]
pub struct GlobalOperand<T: Numeric> {
    pub(crate) backing: Backing<T>,
    /// The buffer's physical extents and strides in scalars, one per physical axis.
    pub geometry: RuntimeGeometry,
    /// The binding's own line width, on top of which a packed operand serves
    /// `packing.factor()` values per stored element.
    #[cube(comptime)]
    pub bound_width: usize,
    /// The kernel's space; the tile's own is its projection onto the spec's axes.
    #[cube(comptime)]
    pub space: Space,
    #[cube(comptime)]
    pub spec: TileSpec,
    /// What a write does to the cell it lands on.
    #[cube(comptime)]
    pub write: Write,
    pub(crate) quant: ComptimeOption<QuantInfo>,
    /// One per [`Scale::Dynamic`](crate::Scale) term and one per [`Divisor::Dynamic`](crate::Divisor)
    /// axis, by physical axis, divisor last; empty for a comptime mapping.
    pub coefficients: Coords<u32>,
    /// One signed value per [`Offset::Dynamic`](crate::Offset) axis; empty for a comptime mapping.
    pub offsets: Coords<i32>,
}

#[cube]
impl<T: Numeric> GlobalOperand<T> {
    /// A launched tensor: the kernel's one `space` projected onto the operand's `spec` axes. The
    /// element type carries the line width, so the served width *is* the binding's. Shape and
    /// strides arrive scalar-unit and convert to lines in the tile.
    pub fn tensor<E: CubePrimitive<Scalar = T>>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> GlobalOperand<T> {
        GlobalOperand::<T>::of_tensor::<E>(
            tensor,
            space,
            spec,
            ComptimeOption::new_None(),
            Coords::<u32>::new(),
            Coords::<i32>::new(),
        )
    }

    /// [`tensor`](Self::tensor) for a gather whose affine map is not all comptime (a runtime
    /// stride, dilation, padding or resize ratio). `coefficients`: one per
    /// [`Scale::Dynamic`](crate::Scale) term and one per [`Divisor::Dynamic`](crate::Divisor) axis,
    /// by physical axis, divisor last. `offsets`: one signed value per
    /// [`Offset::Dynamic`](crate::Offset) axis. Only the lengths are checked, so those orders are
    /// the contract: swap a coefficient for a divisor and the read is silently wrong.
    pub fn gathered<E: CubePrimitive<Scalar = T>>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
    ) -> GlobalOperand<T> {
        GlobalOperand::<T>::of_tensor::<E>(
            tensor,
            space,
            spec,
            ComptimeOption::new_None(),
            coefficients,
            offsets,
        )
    }

    /// [`tensor`](Self::tensor) where the stored element `E` and the served element `T` need not
    /// be the same: a [`packed`](TileSpec::packed) binding holds `u32` words read at `factor`
    /// values each; one stating no packing reads its own element.
    ///
    /// `T` is stated at the call because a packed binding's element is the word, not the value,
    /// so nothing can infer it. Where the binding does read its own element, the two must agree,
    /// which [`tensor`](Self::tensor) proves in the type system and this checks here.
    pub fn stored<E: CubePrimitive>(
        values: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> GlobalOperand<T> {
        let stored = elem_type_of::<E>();
        let read = elem_type_of::<T>();
        comptime!(assert!(
            spec.packing != Packing::Plain || stored == read,
            "GlobalOperand::stored: a binding that states no packing is read at the element it \
             is bound at, {stored:?}, not {read:?}"
        ));
        GlobalOperand::<T>::of_tensor::<E>(
            values,
            space,
            spec,
            ComptimeOption::new_None(),
            Coords::<u32>::new(),
            Coords::<i32>::new(),
        )
    }

    /// The shared body: `E` is the *binding* element, `T` the served scalar, differing only for
    /// a packed or quantized operand, whose store truly holds `E` and whose read view downcasts
    /// back.
    pub(crate) fn of_tensor<E: CubePrimitive>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
        quant: ComptimeOption<QuantInfo>,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
    ) -> GlobalOperand<T> {
        let rank = comptime!(spec.projection.physical_rank());
        let backing = Backing::<T>::new_Buffer(unsafe {
            tensor
                .as_slice()
                .downcast_unchecked::<T>()
                .as_boxed_unchecked()
        });
        GlobalOperand::<T> {
            backing,
            geometry: RuntimeGeometry::of_tensor::<E>(tensor, rank),
            bound_width: tensor.vector_size(),
            space,
            spec,
            write: comptime!(Write::Replace),
            quant,
            coefficients,
            offsets,
        }
    }

    /// An operand whose values are handed to `sink` instead of stored: the walk a buffer gets,
    /// with only its last step a call. The geometry is *stated* because a destination with no
    /// address has none to read; stating the product's own metadata gives the unfused kernel's
    /// store.
    ///
    /// A sink serves the layout-addressed writes and only those: it cannot be staged into shared
    /// memory, written dense, quantized, filled by a tensor map, or [`packed`](TileSpec::packed),
    /// each of which wants an address rather than a call.
    ///
    /// `write` is what the sink does with a value. [`Accumulate`](Write::Accumulate) lets several
    /// instances write one cell, and requires the buffer behind it to hold the monoid's identity
    /// before the launch. Nothing here can check that: the sink cannot read.
    pub fn sink(
        sink: ErasedTensor<T, WriteOnly>,
        geometry: RuntimeGeometry,
        #[comptime] vector_size: usize,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
        #[comptime] write: Write,
    ) -> GlobalOperand<T> {
        // A packing multiplies a binding's own width, but a sink has only the width it states.
        // Refused here, where the spec says it, rather than left to the width mismatch cubecl
        // reports off the erased tensor.
        comptime!(assert!(
            spec.packing == Packing::Plain,
            "GlobalOperand::sink: a sink is written at the width it states, so its spec may not \
             state a packing ({:?}) on top of it",
            spec.packing
        ));
        GlobalOperand::<T> {
            backing: Backing::<T>::new_WriteCall(sink),
            geometry,
            bound_width: vector_size,
            space,
            spec,
            write,
            quant: ComptimeOption::new_None(),
            coefficients: Coords::<u32>::new(),
            offsets: Coords::<i32>::new(),
        }
    }

    /// An operand whose values come from `source` instead of from memory: the fuse-on-read twin
    /// of [`sink`](Self::sink), stated geometry and all. A source serves layout-addressed reads
    /// only: no staging, dense reads, quantization, tensor maps or [`packed`](TileSpec::packed).
    pub fn source(
        source: ErasedTensor<T, ReadOnly>,
        geometry: RuntimeGeometry,
        #[comptime] vector_size: usize,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> GlobalOperand<T> {
        comptime!(assert!(
            spec.packing == Packing::Plain,
            "GlobalOperand::source: a source is read at the width it states, so its spec may not \
             state a packing ({:?}) on top of it",
            spec.packing
        ));
        GlobalOperand::<T> {
            backing: Backing::<T>::new_ReadCall(source),
            geometry,
            bound_width: vector_size,
            space,
            spec,
            write: comptime!(Write::Replace),
            quant: ComptimeOption::new_None(),
            coefficients: Coords::<u32>::new(),
            offsets: Coords::<i32>::new(),
        }
    }

    /// This operand as a [`Tile`] at the top of `levels`: a memory kind over the operand's own
    /// axes, placed where a kernel argument's tile is.
    pub fn tile(self, #[comptime] levels: Vec<Level>) -> Tile<T> {
        let place = comptime!(Placement::root(
            self.space.subspace(self.spec.axes()),
            levels
        ));
        Tile::new(TileKind::new_Memory(Memory::<T>::global(self)), place)
    }
}

#[cube]
impl<T: Numeric> Memory<T> {
    /// The memory tile a launched operand becomes: the top window boxed over the operand's
    /// physical axes, in global memory.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn global(operand: GlobalOperand<T>) -> Memory<T> {
        let backing = operand.backing;
        let geometry = operand.geometry;
        let quant = operand.quant;
        let coefficients = operand.coefficients;
        let offsets = operand.offsets;
        let bound_width = comptime!(operand.bound_width);
        let space = comptime!(operand.space.clone());
        let spec = comptime!(operand.spec.clone());
        let write = comptime!(operand.write);
        // The one projection: the kernel's space narrowed to this operand's axes. What the
        // instances and units are to these cells is stamped level by level on the way down
        // ([`Memory::at`]): a fresh tile has been distributed out by nothing yet.
        let split_share = comptime!(SplitShare::Whole);
        let space = comptime!(space.subspace(spec.axes()));
        let projection = comptime!(spec.projection.clone());
        // The operand addresses *coordinates*; the buffer's storage tiling is the layout's business
        // ([`positional`] below), and splitting a coordinate into digits is what it does with it.
        let coords = comptime!(projection.untiled());
        // How the buffer holds its values: what a quantized operand's scheme says, else what the
        // spec states. One statement, whichever door minted it, so nothing below asks twice.
        let packing = #[comptime]
        match &quant {
            ComptimeOption::Some(info) => comptime!(scheme_packing(info.scheme)),
            ComptimeOption::None => comptime!(spec.packing),
        };
        // A packed store serves `factor` values per stored element, on top of the binding's own
        // line width; a sub-word store serves its stated width out of one word.
        let vector_size = comptime!(packing.served(bound_width));
        // The runtime lists, checked against the statement here, where they arrive.
        let dims_given = geometry.shape.len();
        let coefficients_given = coefficients.len();
        let offsets_given = offsets.len();
        comptime!(assert!(
            dims_given == projection.physical_rank(),
            "GlobalOperand: the projection addresses {} physical dims but {dims_given} were given",
            projection.physical_rank()
        ));
        comptime!(assert!(
            coefficients_given == coords.dynamic_coefficient_count(),
            "GlobalOperand: the projection has {} Dynamic coefficients and divisors but \
             {coefficients_given} were given",
            coords.dynamic_coefficient_count()
        ));
        comptime!(assert!(
            offsets_given == coords.dynamic_offset_count(),
            "GlobalOperand: the projection has {} Dynamic offsets but {offsets_given} were given",
            coords.dynamic_offset_count()
        ));
        comptime!(check_operand(
            &space,
            &spec,
            &coords,
            quant.is_some(),
            vector_size
        ));
        // Off the projection rather than the space: a gathered operand's buffer has fewer
        // physical axes than its logical space has axes, and a storage-tiled one has more.
        let rank = comptime!(projection.physical_rank());
        let last = comptime!(rank - 1);
        let w = comptime!(vector_size as u32);
        let mut physical_shape = Coords::<u32>::new();
        let mut physical_strides = Coords::<u32>::new();
        #[unroll]
        for i in 0..rank {
            let extent = geometry.shape.at(i);
            let stride = geometry.strides.at(i);
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
        // `BufferLayout`'s own physical-position map: the operand's projection relabeled by position,
        // since the layout is handed coordinates a gather has already resolved. Storage tiling
        // survives it, so `physical_shape` is `[pre…, grid…, …, tile…]` in synthetic-axis order.
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
        Memory::<T> {
            address: comptime!(AddressSpace::Global),
            store: Store::<T> {
                backing,
                vector_size: comptime!(vector_size),
                quant,
                packing: comptime!(packing),
            },
            layout: BufferLayout {
                physical_shape,
                physical_strides,
                projection: gmem_projection,
                rows: RowPlacement::InOrder,
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
                write,
                units: spec.units,
                storage: spec.storage,
            }),
            unit_share: comptime!(UnitShare::Repeated),
            split_share,
            init_from: comptime!(InitFrom::Cell),
            factor: Factor::none(),
            lands: false,
        }
    }
}

/// The contract an operand's statement owes, refused at `of` on the host, the one place the
/// spec, the projection, the space and the served width are all in hand. `coords` is the
/// projection in coordinate space ([`Projection::untiled`]).
fn check_operand(
    space: &Space,
    spec: &TileSpec,
    coords: &Projection,
    quantized: bool,
    vector_size: usize,
) {
    let projection = &spec.projection;
    // The scales are gridded over the operand's *logical* axes, while `at` re-windows them over
    // the physical ones. The two ranks coincide for every direct operand, tiled or not, and
    // diverge under a gather.
    assert!(
        !quantized || coords.is_direct(),
        "GlobalOperand: a gathered operand cannot be quantized; its scale grid is shaped over its \
         logical axes, which its buffer's physical axes no longer match"
    );
    assert!(
        !quantized || spec.packing == Packing::Plain,
        "GlobalOperand: a quantized operand's scheme already states how its values are stored, so \
         its spec may not state a packing too"
    );
    // The operand's own contract, checked here rather than at `TileSpec` construction because it
    // turns on the served width, which only this call knows. `StridedTileSource` already checked
    // a padded stage width for the specs it builds; this catches hand-built ones too.
    projection.validate(vector_size);
    // A `Disjoint` claim is about the axes' extents, and this is the one place the projection
    // and the space are both in hand.
    projection.validate_composition(|axis| space.extent(axis));
    let coord_rank = projection.coordinate_rank();
    assert!(
        spec.boundaries.is_empty() || spec.boundaries.len() == coord_rank,
        "GlobalOperand: boundaries rank ({}) does not match coordinate rank ({coord_rank})",
        spec.boundaries.len()
    );
    // A clamped vector line is only valid if the innermost coordinate axis is not clamped. The
    // source builder derives that per-axis mask; this catches hand-built specs too.
    assert!(
        vector_size == 1 || spec.boundaries.last().copied().flatten() != Some(Boundary::Clamp),
        "GlobalOperand: Boundary::Clamp cannot clamp the vectorized innermost axis (served at \
         {vector_size})"
    );
}

/// [`full_window`] for the top gmem tile, over the *physical* axes, where an axis may be
/// [`Dynamic`](crate::Extent) and read its runtime size from `bound`, so the problem shape never
/// specializes the kernel; a gathered operand always reads `bound`, no one extent sizing its axes.
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

/// Where a gathered physical axis's top window starts and the phase its division left behind:
/// `⌊offset / divisor⌋` and `offset mod divisor`. A rational mapping absorbs only its divisor's
/// multiples, handing the rest to [`ProjectionInKernel`](crate::ProjectionInKernel); an integer one all.
///
/// The floor is the host's whenever both sides are comptime; only a `Dynamic` offset or divisor
/// pays for one in the kernel.
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
                    .cast::<i32>(),
            };
            let (start, residue) = floor_div_rem(offset, divisor);
            (start, residue.cast::<u32>())
        }
    }
}
