//! An operand as a single launch argument: its values, a buffer or a TMA tensor map, its comptime
//! [`TileSpec`], and the scales its values are read under, each a [`TileArg`] of its own.

use std::marker::PhantomData;

use cubecl::frontend::{AsMutExpand, AsRefExpand, CubeDebug, ExpandTypeClone, IntoExpand, IntoMut};
use cubecl::frontend::{buffer_len::expand_buffer_length_native, slice};
use cubecl::ir::pliron::{r#type::TypeHandle, value::Value};
use cubecl::ir::{Scope, interfaces::TypedExt, types::VectorType};
use cubecl::prelude::*;
use cubecl::std::tensor::{
    ViewMut,
    layout::{CoordsDyn, VirtualLayout, VirtualLayoutExpand},
    view::launch::ViewCompilationArg,
};
use cubecl::unexpanded;

use crate::tile::packing::base::Packing;
use crate::*;

/// An operand as a single launch argument: its values ([`Values`]), its comptime [`TileSpec`],
/// and the scales they are read under ([`ScaleArg`]).
///
/// `E` is the element the values are read at. A buffer is declared at what it holds: `E` where
/// the spec states no packing, the `u32` words it packs them in where it does, `width` of them a
/// line. Neither is a type of the kernel's: a tile reads its buffer at `E` and its width from the
/// buffer itself.
pub struct TileArg<'a, E: Numeric> {
    pub values: Values<'a, E>,
    pub spec: TileSpec,
    pub scale: ScaleArg,
}

/// Where a [`TileArg`]'s values are: a buffer, or a TMA tensor map, which a stage lands as it
/// lies, and where a position in the operand's space lands in it.
pub enum Values<'a, E: Numeric> {
    Tensor(&'a Tensor<E>),
    TensorMap {
        map: TensorMap<E, Tiled>,
        layout: VirtualLayout<CoordsDyn, Sequence<i32>>,
    },
}

/// The scales a [`TileArg`]'s values are read under: none, or a coarser operand of its own, which
/// may carry scales in turn. Levels are innermost first, so each covers more than the one inside
/// it: NVFP4's values carry their block scales, which carry the per-tensor scale.
///
/// A level is read at one of two elements, a variant each, since an argument's element is a type:
/// words of block scales, which only the block-scaled instruction reads, or scales read as numbers.
pub enum ScaleArg {
    None,
    Words(Box<TileArg<'static, u32>>),
    Scales(Box<TileArg<'static, f32>>),
}

impl<'a, E: Numeric> TileArg<'a, E> {
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

    /// The buffer as lines of `V`, which must be the width it is bound at: what a write that adds
    /// into a line reads and writes it as.
    pub fn lines<V: Size>(&self) -> &Tensor<Vector<E, V>> {
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
pub struct TileArgExpand<'a, E: Numeric> {
    pub values: ValuesExpand<'a, E>,
    pub spec: TileSpec,
    pub scale: ScaleArgExpand,
}

/// The device side of [`Values`]. A buffer keeps what it was declared as, for
/// [`lines`](TileArg::lines).
pub enum ValuesExpand<'a, E: Numeric> {
    Tensor {
        tensor: &'a TensorExpand<E>,
        buffer: Value,
        meta: TensorMetaExpand,
        width: usize,
    },
    TensorMap {
        map: NativeExpand<TensorMap<E, Tiled>>,
        layout: VirtualLayoutExpand<CoordsDyn, Sequence<i32>>,
    },
}

/// The device side of a [`ScaleArg`].
#[derive(Clone)]
pub enum ScaleArgExpand {
    None,
    Words(Box<TileArgExpand<'static, u32>>),
    Scales(Box<TileArgExpand<'static, f32>>),
}

impl<'a, E: Numeric> TileArgExpand<'a, E> {
    pub fn __expand_tile_method(
        &self,
        scope: &Scope,
        partitioning: &PartitioningExpand,
    ) -> TileExpand {
        let values = self.values_tile(scope, partitioning);
        self.scale
            .scaling(scope, &values, partitioning, self.is_tensor_map())
    }

    pub fn __expand_tile_gathered_method(
        &self,
        scope: &Scope,
        partitioning: &PartitioningExpand,
        coefficients: CoordsExpand<u32>,
        offsets: CoordsExpand<i32>,
    ) -> TileExpand {
        let ValuesExpand::Tensor { tensor, .. } = &self.values else {
            panic!("TileArg::tile_gathered: a gather reads a buffer; a tensor map lands boxes")
        };
        let values = gathered_tile::expand::<E>(
            scope,
            tensor,
            partitioning,
            coefficients,
            offsets,
            self.spec.clone(),
        );
        self.scale.scaling(scope, &values, partitioning, false)
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
        let ValuesExpand::Tensor { tensor, .. } = &self.values else {
            panic!("TileArg::tile_paged: a pool of pages is a buffer; a tensor map lands boxes")
        };
        let values = paged_tile::expand::<E>(scope, tensor, table, partitioning, self.spec.clone());
        self.scale.scaling(scope, &values, partitioning, false)
    }

    pub fn __expand_lines_method<V: Size>(&self, scope: &Scope) -> &'a TensorExpand<Vector<E, V>> {
        let ValuesExpand::Tensor {
            buffer,
            meta,
            width: bound,
            ..
        } = &self.values
        else {
            panic!("TileArg::lines: a tensor map has no lines to add into")
        };
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
        match &self.values {
            ValuesExpand::Tensor { tensor, .. } => {
                buffer_tile::expand::<E>(scope, tensor, partitioning, self.spec.clone())
            }
            ValuesExpand::TensorMap { map, layout } => tensor_map_tile::expand::<E>(
                scope,
                *map,
                layout.clone(),
                partitioning,
                self.spec.clone(),
            ),
        }
    }

    fn is_tensor_map(&self) -> bool {
        matches!(self.values, ValuesExpand::TensorMap { .. })
    }
}

impl ScaleArgExpand {
    /// `values` under these scales, a tile carrying its own scales in turn. A tensor map's values
    /// land in a stage as they lie, so the scales on them are words of block scales, which a stage
    /// keeps beside them; a scale above the words leaves the operand for the sum (`Tile::factor`)
    /// before the stage is filled, and staging checks what is left.
    fn scaling(
        &self,
        scope: &Scope,
        values: &TileExpand,
        partitioning: &PartitioningExpand,
        tensor_map: bool,
    ) -> TileExpand {
        match self {
            ScaleArgExpand::None => values.clone(),
            ScaleArgExpand::Words(level) => {
                values.__expand_mul_method(scope, &level.__expand_tile_method(scope, partitioning))
            }
            ScaleArgExpand::Scales(level) => {
                assert!(
                    !tensor_map,
                    "TileArg: a tensor map lands its values as they lie, so it rides only scales \
                     a stage keeps beside them: words of block scales"
                );
                values.__expand_mul_method(scope, &level.__expand_tile_method(scope, partitioning))
            }
        }
    }
}

/// The host side of a [`TileArg`]: its values, its spec and its scales.
pub struct TileArgLaunch<'a, E: Numeric> {
    values: ValuesLaunch<E>,
    spec: TileSpec,
    scale: ScaleArgLaunch,
    _values: PhantomData<&'a E>,
}

/// The host side of [`Values`]: a tensor and the width it is bound at, or a tensor map's view,
/// boxed for being several times a tensor's size.
enum ValuesLaunch<E: Numeric> {
    Tensor { tensor: TensorArg, width: usize },
    TensorMap(Box<<ViewMut<'static, E, CoordsDyn> as LaunchArg>::RuntimeArg>),
}

/// The host side of a [`ScaleArg`].
pub enum ScaleArgLaunch {
    None,
    Words(Box<TileArgLaunch<'static, u32>>),
    Scales(Box<TileArgLaunch<'static, f32>>),
}

impl<'a, E: Numeric> TileArgLaunch<'a, E> {
    /// `tensor`, bound `width` wide, as `spec` reads it, under no scales.
    pub fn new(tensor: TensorArg, width: usize, spec: TileSpec) -> Self {
        Self::of(ValuesLaunch::Tensor { tensor, width }, spec)
    }

    /// A tensor map's `view`, as `spec` reads it, under no scales.
    pub(crate) fn of_view(
        view: <ViewMut<'static, E, CoordsDyn> as LaunchArg>::RuntimeArg,
        spec: TileSpec,
    ) -> Self {
        Self::of(ValuesLaunch::TensorMap(Box::new(view)), spec)
    }

    fn of(values: ValuesLaunch<E>, spec: TileSpec) -> Self {
        Self {
            values,
            spec,
            scale: ScaleArgLaunch::None,
            _values: PhantomData,
        }
    }

    /// These values under `scale`'s words of block scales, outside the scales they carry already.
    pub fn words(self, scale: TileArgLaunch<'static, u32>) -> Self {
        self.under(|scales| ScaleArgLaunch::Words(Box::new(scales)), scale)
    }

    /// These values under `scale`'s scales, read as numbers, outside the scales they carry
    /// already.
    pub fn scales(self, scale: TileArgLaunch<'static, f32>) -> Self {
        self.under(|scales| ScaleArgLaunch::Scales(Box::new(scales)), scale)
    }

    /// This chain with `level(scale)` as its new outermost level.
    fn under<S: Numeric>(
        mut self,
        level: impl FnOnce(TileArgLaunch<'static, S>) -> ScaleArgLaunch,
        scale: TileArgLaunch<'static, S>,
    ) -> Self {
        self.scale = match self.scale {
            ScaleArgLaunch::None => level(scale),
            ScaleArgLaunch::Words(inner) => {
                ScaleArgLaunch::Words(Box::new(inner.under(level, scale)))
            }
            ScaleArgLaunch::Scales(inner) => {
                ScaleArgLaunch::Scales(Box::new(inner.under(level, scale)))
            }
        };
        self
    }
}

/// What a [`TileArg`] compiles to: its values', its spec, and its scales'.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TileArgCompilationArg {
    values: ValuesCompilationArg,
    spec: TileSpec,
    scale: ScaleArgCompilationArg,
}

/// What [`Values`] compile to: a buffer's and the width it is bound at, or a view's.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
enum ValuesCompilationArg {
    Tensor {
        tensor: TensorCompilationArg,
        width: usize,
    },
    TensorMap(ViewCompilationArg<CoordsDyn>),
}

/// What a [`ScaleArg`] compiles to.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum ScaleArgCompilationArg {
    None,
    Words(Box<TileArgCompilationArg>),
    Scales(Box<TileArgCompilationArg>),
}

/// The type a buffer holds: `E`, or the `u32` words a packed binding packs it in.
fn stored<E: Numeric>(scope: &Scope, spec: &TileSpec) -> TypeHandle {
    match spec.packing {
        Packing::Plain => E::__expand_as_type(scope),
        Packing::Packed { .. } => u32::__expand_as_type(scope),
    }
}

impl<'a, E: Numeric> CubeType for TileArg<'a, E> {
    type ExpandType = TileArgExpand<'a, E>;
}

impl CubeType for ScaleArg {
    type ExpandType = ScaleArgExpand;
}

impl<E: Numeric> LaunchArg for TileArg<'static, E> {
    type RuntimeArg = TileArgLaunch<'static, E>;
    type CompilationArg = TileArgCompilationArg;

    fn register(arg: Self::RuntimeArg, launcher: &mut KernelLauncher) -> Self::CompilationArg {
        let TileArgLaunch {
            values,
            spec,
            scale,
            ..
        } = arg;
        let values = match values {
            ValuesLaunch::Tensor { tensor, width } => {
                let line_size = launcher
                    .with_scope(|scope| stored::<E>(scope, &spec).size(scope.ctx()))
                    * width;
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
            ValuesLaunch::TensorMap(view) => {
                ValuesCompilationArg::TensorMap(ViewMut::<'static, E, CoordsDyn>::register(
                    *view, launcher,
                ))
            }
        };
        let scale = ScaleArg::register(scale, launcher);
        TileArgCompilationArg {
            values,
            spec,
            scale,
        }
    }

    fn expand(
        arg: &Self::CompilationArg,
        builder: &mut KernelBuilder,
    ) -> TileArgExpand<'static, E> {
        let values = match &arg.values {
            ValuesCompilationArg::Tensor { tensor, width } => {
                let buffer = match tensor.buffer.inplace {
                    Some(id) => builder.inplace(id),
                    None => {
                        let scope = &builder.scope;
                        let held = stored::<E>(scope, &arg.spec);
                        let line = match *width {
                            1 => held,
                            width => VectorType::get(scope.ctx(), held, width).into(),
                        };
                        builder.tensor(line)
                    }
                };
                let meta = TensorMeta::expand(&tensor.meta, builder);
                let scope = &builder.scope;
                let len = expand_buffer_length_native(scope, buffer);
                let values = slice::from_raw_parts::<E>(
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
            // The map and its layout, as `ViewMut`'s expand takes them apart, kept apart: a tile
            // builds the view where a box moves, at the element it names.
            ValuesCompilationArg::TensorMap(ViewCompilationArg::TensorMapTiled {
                buffer,
                layout,
            }) => {
                let scope = &builder.scope;
                let ty =
                    Type::new(E::elem_type(scope)).with_vector_size(E::__expand_vector_size(scope));
                ValuesExpand::TensorMap {
                    layout: layout.expand(ty, builder),
                    map: <TensorMap<E, Tiled> as LaunchArg>::expand(buffer, builder),
                }
            }
            ValuesCompilationArg::TensorMap(_) => {
                panic!("TileArg: a tensor map's values are tiled boxes")
            }
        };
        let scale = ScaleArg::expand(&arg.scale, builder);
        TileArgExpand {
            values,
            spec: arg.spec.clone(),
            scale,
        }
    }
}

impl LaunchArg for ScaleArg {
    type RuntimeArg = ScaleArgLaunch;
    type CompilationArg = ScaleArgCompilationArg;

    fn register(arg: ScaleArgLaunch, launcher: &mut KernelLauncher) -> ScaleArgCompilationArg {
        match arg {
            ScaleArgLaunch::None => ScaleArgCompilationArg::None,
            ScaleArgLaunch::Words(level) => {
                ScaleArgCompilationArg::Words(Box::new(TileArg::<'static, u32>::register(
                    *level, launcher,
                )))
            }
            ScaleArgLaunch::Scales(level) => {
                ScaleArgCompilationArg::Scales(Box::new(TileArg::<'static, f32>::register(
                    *level, launcher,
                )))
            }
        }
    }

    fn expand(arg: &ScaleArgCompilationArg, builder: &mut KernelBuilder) -> ScaleArgExpand {
        match arg {
            ScaleArgCompilationArg::None => ScaleArgExpand::None,
            ScaleArgCompilationArg::Words(level) => {
                ScaleArgExpand::Words(Box::new(TileArg::<'static, u32>::expand(level, builder)))
            }
            ScaleArgCompilationArg::Scales(level) => {
                ScaleArgExpand::Scales(Box::new(TileArg::<'static, f32>::expand(level, builder)))
            }
        }
    }
}

impl<'a, E: Numeric> Clone for TileArgExpand<'a, E> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            spec: self.spec.clone(),
            scale: self.scale.clone(),
        }
    }
}

impl<'a, E: Numeric> Clone for ValuesExpand<'a, E> {
    fn clone(&self) -> Self {
        match self {
            ValuesExpand::Tensor {
                tensor,
                buffer,
                meta,
                width,
            } => ValuesExpand::Tensor {
                tensor: *tensor,
                buffer: *buffer,
                meta: *meta,
                width: *width,
            },
            ValuesExpand::TensorMap { map, layout } => ValuesExpand::TensorMap {
                map: *map,
                layout: layout.clone(),
            },
        }
    }
}

macro_rules! expand_value {
    ($ty: ty $(, $lt: lifetime, $param: ident)?) => {
        impl$(<$lt, $param: Numeric>)? IntoExpand for $ty {
            type Expand = Self;
            fn into_expand(self, _: &Scope) -> Self {
                self
            }
        }
        impl$(<$lt, $param: Numeric>)? ExpandTypeClone for $ty {
            fn clone_unchecked(&self) -> Self {
                self.clone()
            }
        }
        impl$(<$lt, $param: Numeric>)? IntoMut for $ty {
            fn into_mut(self, _: &Scope) -> Self {
                self
            }
        }
        impl$(<$lt, $param: Numeric>)? CubeDebug for $ty {}
        impl$(<$lt, $param: Numeric>)? AsRefExpand for $ty {
            fn __expand_ref_method(&self, _: &Scope) -> &Self {
                self
            }
        }
        impl$(<$lt, $param: Numeric>)? AsMutExpand for $ty {
            fn __expand_ref_mut_method(&mut self, _: &Scope) -> &mut Self {
                self
            }
        }
    };
}

expand_value!(TileArgExpand<'a, E>, 'a, E);
expand_value!(ScaleArgExpand);
