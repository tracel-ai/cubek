//! A launched memory operand ([`GlobalOperand`]) and the [`Memory`] tile it becomes.

use cubecl::{
    prelude::*,
    std::tensor::{ErasedTensor, WriteOnly},
};

use crate::*;

/// A launched memory operand: a buffer, a fused sink or a fused producer, with its geometry.
#[derive(CubeType)]
pub struct GlobalOperand<T: Numeric> {
    pub(crate) backing: Backing<T>,
    /// The buffer's physical extents and strides in scalars, one per physical axis.
    pub geometry: RuntimeGeometry,
    /// The binding's own line width, before any packing factor.
    #[cube(comptime)]
    pub bound_width: usize,
    /// The kernel's space.
    #[cube(comptime)]
    pub space: Space,
    #[cube(comptime)]
    pub spec: TileSpec,
    /// What a write does to the cell it lands on.
    #[cube(comptime)]
    pub write: Write,
    /// One per [`Scale::Dynamic`](crate::Scale) term and one per
    /// [`Divisor::Dynamic`](crate::Divisor) axis, by physical axis, divisor last.
    pub coefficients: Coords<u32>,
    /// One signed value per [`Offset::Dynamic`](crate::Offset) axis.
    pub offsets: Coords<i32>,
}

#[cube]
impl<T: Numeric> GlobalOperand<T> {
    /// A launched tensor, `space` projected onto the `spec` axes; shape and strides in scalars.
    pub fn tensor<E: CubePrimitive<Scalar = T>>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
    ) -> GlobalOperand<T> {
        GlobalOperand::<T>::of_tensor::<E>(
            tensor,
            space,
            spec,
            Coords::<u32>::new(),
            Coords::<i32>::new(),
        )
    }

    /// [`tensor`](GlobalOperand::tensor) for a gather with a runtime affine map.
    /// `coefficients` and `offsets` follow the field order; only their lengths are checked.
    pub fn gathered<E: CubePrimitive<Scalar = T>>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
        coefficients: Coords<u32>,
        offsets: Coords<i32>,
    ) -> GlobalOperand<T> {
        GlobalOperand::<T>::of_tensor::<E>(tensor, space, spec, coefficients, offsets)
    }

    /// [`tensor`](GlobalOperand::tensor) where stored `E` may differ from served `T`
    /// (a [`packed`](TileSpec::packed) binding).
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
            Coords::<u32>::new(),
            Coords::<i32>::new(),
        )
    }

    /// The shared body: `E` is the binding element, `T` the served scalar.
    pub(crate) fn of_tensor<E: CubePrimitive>(
        tensor: &Tensor<E>,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
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
            coefficients,
            offsets,
        }
    }

    /// An operand whose values are handed to `sink` instead of stored, under the stated geometry.
    /// Under [`Accumulate`](Write::Accumulate) the buffer must hold the monoid's identity.
    pub fn sink(
        sink: ErasedTensor<T, WriteOnly>,
        geometry: RuntimeGeometry,
        #[comptime] vector_size: usize,
        #[comptime] space: Space,
        #[comptime] spec: TileSpec,
        #[comptime] write: Write,
    ) -> GlobalOperand<T> {
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
            coefficients: Coords::<u32>::new(),
            offsets: Coords::<i32>::new(),
        }
    }

    /// An operand whose values come from `source` instead of memory, under the stated geometry.
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
            coefficients: Coords::<u32>::new(),
            offsets: Coords::<i32>::new(),
        }
    }

    /// This operand as a [`Tile`] at the top of `levels`.
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
    /// The memory tile a launched operand becomes, its top window boxed over the physical axes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn global(operand: GlobalOperand<T>) -> Memory<T> {
        let backing = operand.backing;
        let geometry = operand.geometry;
        let coefficients = operand.coefficients;
        let offsets = operand.offsets;
        let bound_width = comptime!(operand.bound_width);
        let space = comptime!(operand.space.clone());
        let spec = comptime!(operand.spec.clone());
        let write = comptime!(operand.write);
        let split_share = comptime!(SplitShare::Whole);
        let space = comptime!(space.subspace(spec.axes()));
        let projection = comptime!(spec.projection.clone());
        let coords = comptime!(projection.untiled());
        let packing = comptime!(spec.packing);
        let vector_size = comptime!(packing.served(bound_width));
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
        comptime!(check_operand(&space, &spec, vector_size));
        // Gathered buffers have fewer physical axes than logical; storage-tiled ones have more.
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
                physical_shape.push(extent / w);
                physical_strides.push(stride);
            } else {
                physical_shape.push(extent);
                physical_strides.push(stride / w);
            }
        }
        let gmem_projection = comptime!(projection.positional());
        // Folded from the physical shape, so tiled (padded) operands get the logical extent.
        let bound = logical_extent(comptime!(gmem_projection.clone()), &physical_shape);
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
                packing: comptime!(packing),
            },
            layout: BufferLayout {
                physical_shape,
                physical_strides,
                projection: gmem_projection,
                rows: RowArrangement::InOrder,
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
            codebook: Codebook::none(),
            lands: false,
        }
    }
}

/// The contract an operand's statement owes, checked on the host.
fn check_operand(space: &Space, spec: &TileSpec, vector_size: usize) {
    let projection = &spec.projection;
    projection.validate(vector_size);
    projection.validate_composition(|axis| space.extent(axis));
    let coord_rank = projection.coordinate_rank();
    assert!(
        spec.boundaries.is_empty() || spec.boundaries.len() == coord_rank,
        "GlobalOperand: boundaries rank ({}) does not match coordinate rank ({coord_rank})",
        spec.boundaries.len()
    );
    assert!(
        vector_size == 1 || spec.boundaries.last().copied().flatten() != Some(Boundary::Clamp),
        "GlobalOperand: Boundary::Clamp cannot clamp the vectorized innermost axis (served at \
         {vector_size})"
    );
}

/// [`full_window`] for the top gmem tile over the physical axes; `Dynamic` axes read `bound`.
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

/// Where a gathered physical axis's top window starts, and the phase its division left behind.
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
        // Floor division: a negative (padding) offset must not round toward zero.
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
