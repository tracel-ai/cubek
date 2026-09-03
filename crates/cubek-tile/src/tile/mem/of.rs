//! Building a memory-backed [`Tile`] from what the launch bound: a tensor, a fused sink, or a
//! fused producer. All of them land in `of_impl`, which boxes the top window over the operand's
//! physical axes.

use cubecl::{
    prelude::*,
    quant::scheme::QuantScheme,
    std::quant::view::KnownScale,
    std::tensor::{ErasedTensor, WriteOnly},
};

use crate::*;

#[cube]
impl<T: Numeric> Tile<T> {
    /// Construct a whole `Gmem` tile straight from a launched tensor: the kernel's one `space`
    /// projected onto the operand's `spec` axes, so no operand carries its own copy. The element
    /// type carries the line width, so the served width *is* the binding's by construction and is
    /// never re-lined in-kernel. Shape/strides arrive scalar-unit and convert to line-unit here.
    pub fn of<E: CubePrimitive<Scalar = T>>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> Tile<T> {
        Tile::<T>::of_tensor::<E>(
            tensor,
            space,
            spec,
            ComptimeOption::new_None(),
            Coords::<u32>::new(),
            Coords::<i32>::new(),
        )
    }

    /// [`of`](Tile::of) for a gather whose affine map is not all comptime: `coefficients` holds one
    /// value per [`Scale::Dynamic`](crate::Scale) term and one per
    /// [`Divisor::Dynamic`](crate::Divisor) axis, interleaved physical axis major so an axis's
    /// divisor follows its own coefficients; `offsets` one signed value per
    /// [`Offset::Dynamic`](crate::Offset) axis. A runtime stride, dilation, padding or resize
    /// ratio is exactly this. Only the lengths are checked, so those index orders are the
    /// contract: swap a coefficient for a divisor and the read is silently wrong.
    pub(crate) fn of_gathered<E: CubePrimitive<Scalar = T>>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
    ) -> Tile<T> {
        Tile::<T>::of_tensor::<E>(
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
    /// read, so the served width is the binding's × that factor. No scales and no scheme: an
    /// operand that also has scales names them as its own tensor.
    pub(crate) fn of_packed<E: CubePrimitive>(
        values: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> Tile<T> {
        comptime!(assert!(
            spec.packing != Packing::Plain,
            "Tile::of_packed: the operand states no packing, so it is a plain tile (Tile::of)"
        ));
        Tile::<T>::of_tensor::<E>(
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
    pub(crate) fn of_dequant<E: CubePrimitive>(
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
        Tile::<T>::of_tensor::<E>(
            values,
            space,
            spec,
            ComptimeOption::new_Some(info),
            Coords::<u32>::new(),
            Coords::<i32>::new(),
        )
    }

    /// Shared body of [`of`](Tile::of)/[`of_dequant`](Tile::of_dequant): `E` is the *binding*
    /// element, `T` the served scalar, differing only for a quantized operand whose served width
    /// is the binding's × the packing factor. Re-typing the buffer to `T` is a static coercion for
    /// a plain operand; a quantized store truly holds the stored type and the read view downcasts
    /// back ([`lines_storage`](MemData::lines_storage)).
    fn of_tensor<E: CubePrimitive>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
        quant: ComptimeOption<QuantInfo>,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
    ) -> Tile<T> {
        let rank = comptime!(spec.projection.physical_rank());
        let backing = Backing::<T>::new_Buffer(unsafe {
            tensor
                .as_slice()
                .downcast_unchecked::<T>()
                .as_boxed_unchecked()
        });
        Tile::<T>::of_impl(
            backing,
            RuntimeGeometry::of_tensor::<E>(tensor, rank),
            tensor.vector_size(),
            space,
            spec,
            Write::Replace,
            quant,
            coefficients,
            offsets,
        )
    }

    /// A tile whose values are handed to `sink` instead of stored: the walk a buffer gets, with
    /// only its last step a call rather than a store. The geometry is *stated* because a
    /// destination with no address has none to read, so a caller that states the product's own
    /// metadata gets the store the unfused kernel would have made.
    ///
    /// A sink serves the layout-addressed writes and only those: it cannot be staged into shared
    /// memory, written dense, quantized, filled by a tensor map, or [`packed`](TileSpec::packed),
    /// each of which wants an address rather than a call.
    ///
    /// `write` is what the sink does with a value. [`Accumulate`](Write::Accumulate) lets several
    /// instances write one cell, and requires the buffer behind it to hold the monoid's identity
    /// before the launch. Nothing here can check that: the sink cannot read.
    pub fn of_sink(
        sink: ErasedTensor<T, WriteOnly>,
        geometry: RuntimeGeometry,
        #[comptime] vector_size: usize,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
        #[comptime] write: Write,
    ) -> Tile<T> {
        // A bound operand reads its width off its binding, so a packing multiplying it is a
        // fact about the two together; a sink has only what it states, and `of_impl` would
        // address it at `vector_size * factor`. Refused here, where the spec says it, rather
        // than left to the width mismatch cubecl reports off the erased tensor.
        comptime!(assert!(
            spec.packing == Packing::Plain,
            "Tile::of_sink: a sink is written at the width it states, so its spec may not \
             state a packing ({:?}) on top of it",
            spec.packing
        ));
        Tile::<T>::of_impl(
            Backing::<T>::new_WriteCall(sink),
            geometry,
            vector_size,
            space,
            spec,
            write,
            ComptimeOption::new_None(),
            Coords::<u32>::new(),
            Coords::<i32>::new(),
        )
    }

    /// A tile whose values come from `source` instead of from memory: the fuse-on-read twin of
    /// [`of_sink`](Tile::of_sink), stated geometry and all. A source serves the layout-addressed
    /// reads and only those: it cannot be staged into shared memory, read dense, quantized, loaded
    /// by a tensor map, or [`packed`](TileSpec::packed).
    pub fn of_source(
        source: ErasedTensor<T, ReadOnly>,
        geometry: RuntimeGeometry,
        #[comptime] vector_size: usize,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> Tile<T> {
        comptime!(assert!(
            spec.packing == Packing::Plain,
            "Tile::of_source: a source is read at the width it states, so its spec may not \
             state a packing ({:?}) on top of it",
            spec.packing
        ));
        Tile::<T>::of_impl(
            Backing::<T>::new_ReadCall(source),
            geometry,
            vector_size,
            space,
            spec,
            Write::Replace,
            ComptimeOption::new_None(),
            Coords::<u32>::new(),
            Coords::<i32>::new(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn of_impl(
        backing: Backing<T>,
        // `geometry` is the destination's physical extents and strides in scalars,
        // one per physical axis; `bound_width` the binding's own line width, on top
        // of which a packed destination serves `packing.factor()` values per stored
        // element.
        geometry: RuntimeGeometry,
        #[comptime] bound_width: usize,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
        #[comptime] write: Write,
        quant: ComptimeOption<QuantInfo>,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
    ) -> Tile<T> {
        // Asked before the projection, which is what drops the contracted axis: only the whole
        // space still has the extent that says whether that axis is split or merely cut.
        let split_share = comptime!(space.split_share_of(spec.axes()));
        let lane_work = comptime!(space.lane_work());
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
        // Free for a bound operand, which builds its geometry off the projection's own rank; the
        // check is for a *stated* one ([`of_sink`](Tile::of_sink)), where too few dims panic on an
        // opaque `Sequence` index below and too many silently ignore their tail.
        let dims_given = geometry.shape.len();
        let physical_rank = comptime!(projection.physical_rank());
        comptime!(assert!(
            dims_given == physical_rank,
            "Tile::of: the projection addresses {physical_rank} physical dims but {dims_given} \
             were given"
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
        // A packed store serves `factor` values per stored element, on top of the binding's own
        // line width.
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
        // `physical_rank` above, off the projection rather than the space: a gathered operand's
        // buffer has fewer physical axes than its logical space has axes, and a storage-tiled one
        // has more.
        let rank = physical_rank;
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
                    backing,
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
                    write,
                    stage,
                }),
                lanes: comptime!(Lanes {
                    share: LaneShare::Whole,
                    work: lane_work,
                }),
                split_share,
                init_from: comptime!(InitFrom::Cell),
            }),
            space: comptime!(space),
        }
    }
}

/// [`full_window`] for the top gmem tile, over the *physical* axes, where an axis may be
/// [`Dynamic`](crate::Extent) and read its runtime size from `bound` instead of a comptime
/// constant, so the problem shape never specializes the kernel. A gathered operand always reads
/// `bound`: no single logical extent sizes a physical axis combining several.
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
/// `⌊offset / divisor⌋` and `offset mod divisor`. An integer mapping divides by `1` and absorbs
/// the offset whole; a rational one absorbs only the multiples of its divisor and hands the rest
/// to [`AxisProjection`](crate::AxisProjection). The floor is the host's whenever both sides are
/// comptime; only a `Dynamic` offset or divisor pays for one in the kernel.
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
