//! An operand as a single launch argument: its values, a buffer or a TMA tensor map, its comptime
//! [`TileSpec`], the element it serves them at, and the scales its values are read under, each a
//! [`TileArg`] of its own.

use std::marker::PhantomData;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::frontend::{buffer_len::expand_buffer_length_native, slice};
use cubecl::ir::pliron::value::Value;
use cubecl::ir::{ElemType, Scope, types::VectorType};
use cubecl::prelude::*;
use cubecl::std::tensor::{
    layout::{CoordsDyn, VirtualLayout, VirtualLayoutExpand},
    view::launch::{VirtualViewLayoutCompilationArg, VirtualViewLayoutLaunch},
};
use cubecl::unexpanded;

use crate::tile::packing::base::Packing;
use crate::*;

/// An operand as a single launch argument: its values, a buffer or a TMA tensor map, its comptime
/// [`TileSpec`], the element its buffer holds, and the scales they are read under.
///
/// The element is data the argument carries, stated where the operand is bound
/// ([`TileArgLaunch::new`]), not a type of the kernel's: what the buffer holds, `width` of them a
/// line. A packed buffer holds `u32` words, its spec's packing naming the fields in them; a reader
/// decodes them into the element it asks for ([`Tile::dequantized`]), or `f32` where it asks for
/// none.
///
/// Scales are levels innermost first, so each covers more than the one inside it: NVFP4's values
/// carry their block scales, which carry the per-tensor scale. A plain level holding `u32` is words
/// of block scales, which only the block-scaled instruction reads; any other, a packed one
/// included, is scales read as numbers.
pub struct TileArg<'a> {
    pub spec: TileSpec,
    /// The element the buffer holds.
    pub element: ElemType,
    _values: PhantomData<&'a ()>,
}

impl TileArg<'_> {
    /// Serve the operand as a [`Tile`] over the kernel's one `partitioning`, under its scales.
    pub fn tile(&self, _partitioning: &Partitioning) -> Tile {
        unexpanded!()
    }

    /// [`tile`](TileArg::tile), read through `pages` where the operand is a pool of pages bound
    /// [`paged`](crate::launch::Arg::paged), and where it lies where `pages` is `None`: one
    /// kernel serves either. The launch decides which it compiles by the `pages` it passes, which
    /// must match the binding: a table for a pool bound `paged`, `None` for any other operand;
    /// a mismatch panics as the kernel expands. A table holds at least one entry, which every
    /// window reads; a launch over no sequences or no positions is the caller's to skip.
    pub fn tile_paged(
        &self,
        _partitioning: &Partitioning,
        _pages: ComptimeOption<&Tensor<u32>>,
    ) -> Tile {
        unexpanded!()
    }

    /// [`tile`](TileArg::tile) for a partly runtime gather map, its dynamic values in
    /// [`GlobalOperand::gathered`]'s order. Only a buffer is gathered.
    pub fn tile_gathered(
        &self,
        _partitioning: &Partitioning,
        _coefficients: Coords<u32>,
        _offsets: Coords<i32>,
    ) -> Tile {
        unexpanded!()
    }

    /// The buffer as lines of `V` of `E`, which must be the element it holds and the width it is
    /// bound at: what a write that adds into a line reads and writes it as. A launch that bound
    /// the buffer at another element is refused as the kernel expands.
    pub fn lines<E: Numeric, V: Size>(&self) -> &Tensor<Vector<E, V>> {
        unexpanded!()
    }
}

/// A buffer of values, served at `E`.
#[cube]
fn buffer_tile<E: Numeric>(
    tensor: &Tensor<E>,
    partitioning: &Partitioning,
    #[comptime] spec: TileSpec,
) -> Tile {
    let space = comptime!(partitioning.clone());
    GlobalOperand::tensor(tensor, comptime!(space.space().clone()), spec)
        .tile::<E>(comptime!(space.levels().to_vec()))
}

/// [`buffer_tile`] through a partly runtime gather map.
#[cube]
fn gathered_tile<E: Numeric>(
    tensor: &Tensor<E>,
    partitioning: &Partitioning,
    coefficients: Coords<u32>,
    offsets: Coords<i32>,
    #[comptime] spec: TileSpec,
) -> Tile {
    let space = comptime!(partitioning.clone());
    GlobalOperand::gathered(
        tensor,
        comptime!(space.space().clone()),
        spec,
        coefficients,
        offsets,
    )
    .tile::<E>(comptime!(space.levels().to_vec()))
}

/// [`buffer_tile`] over a pool of pages, read as the sequences `table` names.
#[cube]
fn paged_tile<E: Numeric>(
    tensor: &Tensor<E>,
    table: &Tensor<u32>,
    partitioning: &Partitioning,
    #[comptime] spec: TileSpec,
) -> Tile {
    let space = comptime!(partitioning.clone());
    GlobalOperand::paged(tensor, table, comptime!(space.space().clone()), spec)
        .tile::<E>(comptime!(space.levels().to_vec()))
}

/// A tensor map, served as a TMA-filled tile.
#[cube]
fn tensor_map_tile<E: Numeric>(
    map: TensorMap<E, Tiled>,
    layout: VirtualLayout<CoordsDyn, Sequence<i32>>,
    partitioning: &Partitioning,
    #[comptime] spec: TileSpec,
) -> Tile {
    let space = comptime!(partitioning.clone());
    let own = comptime!(space.space().subspace(spec.axes()));
    let data = TmaData::from_tensor_map(
        map,
        layout,
        comptime!(own.rank()),
        comptime!(spec.units),
        comptime!(spec.packing),
    );
    Tile::new::<E>(
        TileKind::new_TmaGmem(data),
        comptime!(Placement::root(own.clone(), space.levels().to_vec())),
    )
}

/// The device side of a [`TileArg`].
#[derive(Clone)]
pub struct TileArgExpand<'a> {
    values: ValuesExpand<'a>,
    pub spec: TileSpec,
    /// The element the buffer holds.
    pub element: ElemType,
    /// The scales the values are read under: none, or a coarser operand carrying its own.
    scale: Option<Box<TileArgExpand<'static>>>,
}

/// The device side of the values: a buffer, kept also as it was declared for
/// [`lines`](TileArg::lines), or a tensor map and the layout landing a position in it.
#[derive(Clone)]
enum ValuesExpand<'a> {
    Tensor {
        /// The values, kept at [`Held`] and named at their element where a tile reads them.
        tensor: &'a TensorExpand<Held>,
        buffer: Value,
        meta: TensorMetaExpand,
        width: usize,
    },
    TensorMap {
        map: Value,
        layout: VirtualLayoutExpand<CoordsDyn, Sequence<i32>>,
    },
}

impl<'a> TileArgExpand<'a> {
    pub fn __expand_tile_method(
        &self,
        scope: &Scope,
        partitioning: &PartitioningExpand,
    ) -> TileExpand {
        let values = self.values_tile(scope, partitioning);
        self.scaling(scope, values, partitioning)
    }

    pub fn __expand_tile_gathered_method(
        &self,
        scope: &Scope,
        partitioning: &PartitioningExpand,
        coefficients: CoordsExpand<u32>,
        offsets: CoordsExpand<i32>,
    ) -> TileExpand {
        let ValuesExpand::Tensor { tensor, meta, .. } = &self.values else {
            panic!("TileArg::tile_gathered: a gather reads a buffer; a tensor map lands boxes")
        };
        let _held = scope.define_type::<Held>(self.element);
        let tensor = scope.create_kernel_ref(TensorExpand::<Held>::__expand_from_parts(
            *meta,
            tensor.expand.into(),
        ));
        let values = gathered_tile::expand::<Held>(
            scope,
            tensor,
            partitioning,
            coefficients,
            offsets,
            self.spec.clone(),
        );
        self.scaling(scope, values, partitioning)
    }

    pub fn __expand_tile_paged_method(
        &self,
        scope: &Scope,
        partitioning: &PartitioningExpand,
        pages: ComptimeOptionExpand<&Tensor<u32>>,
    ) -> TileExpand {
        let ComptimeOptionExpand::Some(table) = pages else {
            return self.__expand_tile_method(scope, partitioning);
        };
        let ValuesExpand::Tensor { tensor, meta, .. } = &self.values else {
            panic!("TileArg::tile_paged: a pool of pages is a buffer; a tensor map lands boxes")
        };
        let _held = scope.define_type::<Held>(self.element);
        let tensor = scope.create_kernel_ref(TensorExpand::<Held>::__expand_from_parts(
            *meta,
            tensor.expand.into(),
        ));
        let values =
            paged_tile::expand::<Held>(scope, tensor, table, partitioning, self.spec.clone());
        self.scaling(scope, values, partitioning)
    }

    pub fn __expand_lines_method<E: Numeric, V: Size>(
        &self,
        scope: &Scope,
    ) -> &'a TensorExpand<Vector<E, V>> {
        let ValuesExpand::Tensor {
            buffer,
            meta,
            width: bound,
            ..
        } = &self.values
        else {
            panic!("TileArg::lines: a tensor map has no lines to add into")
        };
        let named = E::elem_type(scope);
        assert_eq!(
            named, self.element,
            "TileArg::lines: the buffer is bound at {}, not {named}",
            self.element
        );
        let width = V::__expand_value(scope);
        assert_eq!(
            width, *bound,
            "TileArg::lines: the buffer is bound {bound} wide, not {width}"
        );
        let len = expand_buffer_length_native(scope, *buffer);
        let lines = slice::from_raw_parts::<Vector<E, V>>(
            scope,
            *buffer,
            0usize.into_expand(scope),
            len.into(),
        );
        scope.create_kernel_ref(TensorExpand::__expand_from_parts(*meta, lines))
    }

    /// The values alone, under none of their scales.
    fn values_tile(&self, scope: &Scope, partitioning: &PartitioningExpand) -> TileExpand {
        let _held = scope.define_type::<Held>(self.element);
        match &self.values {
            ValuesExpand::Tensor { tensor, meta, .. } => {
                let tensor = scope.create_kernel_ref(TensorExpand::<Held>::__expand_from_parts(
                    *meta,
                    tensor.expand.into(),
                ));
                buffer_tile::expand::<Held>(scope, tensor, partitioning, self.spec.clone())
            }
            ValuesExpand::TensorMap { map, layout } => tensor_map_tile::expand::<Held>(
                scope,
                (*map).into(),
                layout.clone(),
                partitioning,
                self.spec.clone(),
            ),
        }
    }

    /// `values` under these scales, a tile carrying its own scales in turn. A tensor map's values
    /// land in a stage as they lie, so the scales on them are words of block scales, which a stage
    /// keeps beside them; a scale above the words leaves the operand for the sum (`Tile::factor`)
    /// before the stage is filled, and staging checks what is left.
    fn scaling(
        &self,
        scope: &Scope,
        values: TileExpand,
        partitioning: &PartitioningExpand,
    ) -> TileExpand {
        let Some(level) = &self.scale else {
            return values;
        };
        let words =
            level.element == u32::elem_type_native() && level.spec.packing == Packing::Plain;
        assert!(
            words || !matches!(self.values, ValuesExpand::TensorMap { .. }),
            "TileArg: a tensor map lands its values as they lie, so it rides only scales a stage \
             keeps beside them: words of block scales"
        );
        values.__expand_mul_method(scope, &level.__expand_tile_method(scope, partitioning))
    }
}

/// The host side of a [`TileArg`]: its values, its spec, the element its buffer holds and its
/// scales.
pub struct TileArgLaunch {
    values: ValuesLaunch,
    spec: TileSpec,
    element: ElemType,
    scale: Option<Box<TileArgLaunch>>,
}

/// The host side of the values: a tensor and the width it is bound at, or a tensor map, boxed for
/// being several times a tensor's size, and its layout.
enum ValuesLaunch {
    Tensor {
        tensor: TensorArg,
        width: usize,
    },
    TensorMap {
        map: Box<TensorMapArg<Tiled>>,
        layout: VirtualViewLayoutLaunch<CoordsDyn, Sequence<i32>, TensorMapArg<Tiled>>,
    },
}

impl TileArgLaunch {
    /// `tensor`, holding `element`, bound `width` wide, as `spec` reads it, under no scales.
    pub fn new(tensor: TensorArg, element: ElemType, width: usize, spec: TileSpec) -> Self {
        Self::of(ValuesLaunch::Tensor { tensor, width }, element, spec)
    }

    /// A tensor map landing through `layout`, holding the element its descriptor moves, as `spec`
    /// reads it, under no scales.
    pub(crate) fn of_tensor_map(
        map: TensorMapArg<Tiled>,
        layout: VirtualViewLayoutLaunch<CoordsDyn, Sequence<i32>, TensorMapArg<Tiled>>,
        spec: TileSpec,
    ) -> Self {
        let element = map.metadata.elem_ty;
        let map = Box::new(map);
        Self::of(ValuesLaunch::TensorMap { map, layout }, element, spec)
    }

    fn of(values: ValuesLaunch, element: ElemType, spec: TileSpec) -> Self {
        Self {
            values,
            spec,
            element,
            scale: None,
        }
    }

    /// These values under `scale`, outside the scales they carry already: words of block scales
    /// where `scale` is a plain level holding `u32`, scales read as numbers otherwise.
    pub fn scales(mut self, scale: TileArgLaunch) -> Self {
        self.scale = Some(Box::new(match self.scale {
            None => scale,
            Some(inner) => inner.scales(scale),
        }));
        self
    }
}

/// What a [`TileArg`] compiles to: its values', its spec, its element and its scales'.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TileArgCompilationArg {
    values: ValuesCompilationArg,
    spec: TileSpec,
    element: ElemType,
    scale: Option<Box<TileArgCompilationArg>>,
}

/// What the values compile to: a buffer's and the width it is bound at, or a tensor map's layout.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum ValuesCompilationArg {
    Tensor {
        tensor: TensorCompilationArg,
        width: usize,
    },
    TensorMap(VirtualViewLayoutCompilationArg<CoordsDyn, Sequence<i32>>),
}

impl<'a> CubeType for TileArg<'a> {
    type ExpandType = TileArgExpand<'a>;
}

impl LaunchArg for TileArg<'static> {
    type RuntimeArg = TileArgLaunch;
    type CompilationArg = TileArgCompilationArg;

    fn register(arg: Self::RuntimeArg, launcher: &mut KernelLauncher) -> Self::CompilationArg {
        let TileArgLaunch {
            values,
            spec,
            element,
            scale,
        } = arg;
        let values = match values {
            ValuesLaunch::Tensor { tensor, width } => {
                let line_size = element.size() * width;
                let meta = TensorMetaLaunch::new(tensor.size() / width, tensor.shape().len());
                let buffer = match &tensor {
                    TensorArg::Handle { .. } => BufferCompilationArg { inplace: None },
                    TensorArg::Alias { input_pos, .. } => BufferCompilationArg {
                        inplace: Some(*input_pos),
                    },
                };
                launcher.register_tensor(tensor, line_size);
                let meta = TensorMeta::register(meta, launcher);
                ValuesCompilationArg::Tensor {
                    tensor: TensorCompilationArg { meta, buffer },
                    width,
                }
            }
            ValuesLaunch::TensorMap { map, layout } => {
                let layout = layout.register(&map, Type::new(element), launcher);
                launcher.register_tensor_map(*map, element.size());
                ValuesCompilationArg::TensorMap(layout)
            }
        };
        let scale = scale.map(|level| Box::new(TileArg::register(*level, launcher)));
        TileArgCompilationArg {
            values,
            spec,
            element,
            scale,
        }
    }

    fn expand(arg: &Self::CompilationArg, builder: &mut KernelBuilder) -> TileArgExpand<'static> {
        let values = match &arg.values {
            ValuesCompilationArg::Tensor { tensor, width } => {
                let buffer = match tensor.buffer.inplace {
                    Some(id) => builder.inplace(id),
                    None => {
                        let scope = &builder.scope;
                        let held = arg.element.to_type(scope.ctx());
                        let line = match *width {
                            1 => held,
                            width => VectorType::get(scope.ctx(), held, width).into(),
                        };
                        builder.tensor(line)
                    }
                };
                let meta = TensorMeta::expand(&tensor.meta, builder);
                let scope = &builder.scope;
                let _held = scope.define_type::<Held>(arg.element);
                let len = expand_buffer_length_native(scope, buffer);
                let values = slice::from_raw_parts::<Held>(
                    scope,
                    buffer,
                    0usize.into_expand(scope),
                    len.into(),
                );
                ValuesExpand::Tensor {
                    tensor: scope
                        .create_kernel_ref(TensorExpand::__expand_from_parts(meta, values)),
                    buffer,
                    meta,
                    width: *width,
                }
            }
            // The map and its layout kept apart: a tile builds the view where a box moves, at the
            // element the map serves.
            ValuesCompilationArg::TensorMap(layout) => ValuesExpand::TensorMap {
                layout: layout.expand(Type::new(arg.element), builder),
                map: builder.tensor_map(),
            },
        };
        let scale = arg
            .scale
            .as_ref()
            .map(|level| Box::new(TileArg::expand(level, builder)));
        TileArgExpand {
            values,
            spec: arg.spec.clone(),
            element: arg.element,
            scale,
        }
    }
}

impl IntoExpand for TileArgExpand<'_> {
    type Expand = Self;
    fn into_expand(self, _: &Scope) -> Self {
        self
    }
}

impl ExpandTypeClone for TileArgExpand<'_> {
    fn clone_unchecked(&self) -> Self {
        self.clone()
    }
}

impl IntoMut for TileArgExpand<'_> {
    fn into_mut(self, _: &Scope) -> Self {
        self
    }
}

impl CubeDebug for TileArgExpand<'_> {}

impl AsRefExpand for TileArgExpand<'_> {
    fn __expand_ref_method(&self, _: &Scope) -> &Self {
        self
    }
}

impl AsMutExpand for TileArgExpand<'_> {
    fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
        self
    }
}

#[cfg(test)]
mod tests {
    use cubecl::ir::FloatKind;

    use super::*;

    /// A one-axis buffer of `f32`, expanded as a kernel argument outside any launch.
    fn expanded_f32_buffer(builder: &mut KernelBuilder) -> TileArgExpand<'static> {
        let mut launcher = KernelLauncher::new(builder_settings());
        let meta = TensorMeta::register(TensorMetaLaunch::new(4, 1), &mut launcher);
        let arg = TileArgCompilationArg {
            values: ValuesCompilationArg::Tensor {
                tensor: TensorCompilationArg {
                    meta,
                    buffer: BufferCompilationArg { inplace: None },
                },
                width: 1,
            },
            spec: TileSpec::direct(&[Axis(0)]),
            element: ElemType::Float(FloatKind::F32),
            scale: None,
        };
        TileArg::expand(&arg, builder)
    }

    fn builder_settings() -> KernelSettings {
        KernelSettings::new(
            cubecl::ir::settings::Dim3::new_single(),
            cubecl::ir::settings::ExecutionMode::Checked,
            cubecl::ir::AddressType::U32,
        )
    }

    #[test]
    #[should_panic(expected = "TileArg::lines: the buffer is bound at f32, not i32")]
    fn lines_refuse_a_buffer_bound_at_another_element() {
        let mut builder = KernelBuilder::new(builder_settings());
        let arg = expanded_f32_buffer(&mut builder);
        arg.__expand_lines_method::<i32, Const<1>>(&builder.scope);
    }

    #[test]
    fn lines_read_a_buffer_at_the_element_it_is_bound_at() {
        let mut builder = KernelBuilder::new(builder_settings());
        let arg = expanded_f32_buffer(&mut builder);
        arg.__expand_lines_method::<f32, Const<1>>(&builder.scope);
    }
}
