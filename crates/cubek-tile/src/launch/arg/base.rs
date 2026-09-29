//! The [`Arg`] builder, which turns a launched tensor into an operand.

use core::marker::PhantomData;

use cubecl::prelude::*;

use super::analysis::{Boundaries, Labels, Refusal};
use crate::{
    Axis, Boundary, Field, Geometry, Launcher, LineMisfit, Packing, Projection, Storage,
    StoragePartitioning, TileArgLaunch, TileSpec,
};

/// Typestate marker: the operand's axes are not yet stated.
pub struct Unlabelled;
/// Typestate marker: the operand's axes are stated.
pub struct Labelled;

/// Whether an operand's reads are bounds-checked, and how.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BoundaryPolicy {
    /// Checked with [`Boundary::Zero`] on the axes that can leave the buffer.
    #[default]
    Derived,
    /// Unchecked; the caller guarantees every read is in bounds.
    Unchecked,
    /// Checked with this boundary on every axis that is not provably in bounds.
    Every(Boundary),
}

/// What the builder accumulates.
struct ArgData<'a> {
    launch: &'a Launcher,
    /// The tensor this operand is served from; `None` for a destination with no address.
    binding: Option<TensorBinding>,
    geometry: Geometry,
    axes: &'a [Axis],
    batches: &'a [Axis],
    /// The operand's own affine mapping; `None` derives it from the labelled dims.
    projection: Option<Projection>,
    width: usize,
    boundary: BoundaryPolicy,
    packing: Packing,
    /// Whether the labelled dims bind in stride order ([`in_stride_order`](Arg::in_stride_order)).
    in_stride_order: bool,
}

/// One operand of a launch, being described; `S` says whether its axes are stated yet.
pub struct Arg<'a, S> {
    data: ArgData<'a>,
    _state: PhantomData<S>,
}

impl<'a> Arg<'a, Unlabelled> {
    pub(crate) fn bound(launch: &'a Launcher, binding: TensorBinding) -> Self {
        let geometry = Geometry::from(&binding);
        Self::over(launch, Some(binding), geometry)
    }

    pub(crate) fn unbound(launch: &'a Launcher, geometry: &Geometry) -> Self {
        Self::over(launch, None, geometry.clone())
    }

    fn over(launch: &'a Launcher, binding: Option<TensorBinding>, geometry: Geometry) -> Self {
        Arg {
            data: ArgData {
                launch,
                binding,
                geometry,
                axes: &[],
                batches: &[],
                projection: None,
                width: 1,
                boundary: BoundaryPolicy::Derived,
                packing: Packing::Plain,
                in_stride_order: false,
            },
            _state: PhantomData,
        }
    }

    /// The axes the operand's trailing buffer dims carry.
    pub fn axes(mut self, axes: &'a [Axis]) -> Arg<'a, Labelled> {
        self.data.axes = axes;
        self.labelled()
    }

    /// An explicit affine [`Projection`] for a gathered operand. Overruns past the buffer's tail
    /// are not detected; a gather that overruns must state [`BoundaryPolicy::Every`].
    pub fn gathered(mut self, projection: Projection) -> Arg<'a, Labelled> {
        self.data.projection = Some(projection);
        self.labelled()
    }

    fn labelled(self) -> Arg<'a, Labelled> {
        Arg {
            data: self.data,
            _state: PhantomData,
        }
    }
}

impl<'a> Arg<'a, Labelled> {
    /// The batch axes in the output's order, right-aligned to the leading dims (numpy broadcast).
    pub fn batches(mut self, axes: &'a [Axis]) -> Self {
        self.data.batches = axes;
        self
    }

    /// Bind the labelled dims in the order they step, coarsest first; batch dims stay put.
    pub fn in_stride_order(mut self) -> Self {
        self.data.in_stride_order = true;
        self
    }

    /// Serve the innermost axis in `width`-wide lines; that axis must be contiguous.
    pub fn vectorize(mut self, width: usize) -> Self {
        self.data.width = width;
        self
    }

    /// How this operand's edges are checked.
    pub fn boundary(mut self, policy: BoundaryPolicy) -> Self {
        self.data.boundary = policy;
        self
    }

    /// This operand's values are `field`-wide fields of a stored word.
    pub fn packed(mut self, field: impl Into<Field>) -> Self {
        self.data.packing = Packing::Packed {
            field: field.into(),
        };
        self
    }

    /// The operand, bound; panics with the [`Refusal`] if it cannot be.
    pub fn build(self) -> Bound {
        let (bound, _) = self.realize().unwrap_or_else(|refusal| panic!("{refusal}"));
        bound
    }

    /// [`build`](Self::build) without the tensor argument, for an operand with no address.
    pub fn build_spec(self) -> Unbound {
        let (bound, geometry) = self.realize().unwrap_or_else(|refusal| panic!("{refusal}"));
        Unbound {
            spec: bound.spec,
            vector_size: bound.vector_size,
            geometry,
        }
    }

    /// The derivation both builds share.
    fn realize(self) -> Result<(Bound, Geometry), Refusal> {
        let data = self.data;
        let launch = data.launch;
        let width = data.width;
        let stored = data.binding.as_ref().map(|b| b.tiling).unwrap_or_default();
        let stated = data.projection.is_some() || stored.is_tiled();
        let (geometry, axes) =
            stride_ordered(data.geometry, data.axes, data.in_stride_order, stated)?;
        let storage = storage_of(&geometry, &axes, data.projection.is_none(), launch);
        let labels = match data.projection {
            Some(projection) => {
                Labels::stated(&geometry, launch, projection, &axes, data.batches, stored)?
            }
            None => Labels::new(&geometry, &axes, data.batches)?,
        };
        let Labels {
            geometry,
            projection,
            addressed,
        } = labels;
        projection.validate(width);
        // A stated width is checked here: `stride / width` would truncate silently.
        // A line runs along the innermost dim's axis; a buffer with no labelled dim has none.
        let labels = projection.dense_labels();
        let served = match labels.last() {
            Some(&axis) => geometry.serves(&[(axis, width)], &labels),
            None if width == 1 => Ok(()),
            None => Err(LineMisfit::NoDims),
        };
        served.map_err(|why| Refusal::WidthNotServed { width, why })?;
        let overhangs = launch.overhangs();
        let boundaries = Boundaries::new(
            data.boundary,
            &projection,
            &addressed,
            launch.space(),
            &overhangs,
            width,
        )?;
        let spec = TileSpec {
            projection,
            boundaries: boundaries.modes,
            units: launch.cube_dim().num_elems() as usize,
            packing: data.packing,
            storage,
        };
        let tensor = data
            .binding
            .map(|binding| settled_tensor(binding, &geometry));
        let bound = Bound {
            tensor,
            vector_size: width,
            spec,
        };
        Ok((bound, geometry))
    }
}

/// The labelled dims in stride order where asked; refused for a stated layout.
fn stride_ordered(
    geometry: Geometry,
    axes: &[Axis],
    in_stride_order: bool,
    stated: bool,
) -> Result<(Geometry, Vec<Axis>), Refusal> {
    match in_stride_order {
        true if stated => Err(Refusal::StrideOrderOfAStatedLayout),
        true => Ok(geometry.in_stride_order(axes)),
        false => Ok((geometry, axes.to_vec())),
    }
}

/// The coarsest level whose windows the operand's storage tiles address with one stride per axis
/// ([`Contiguous`](Storage::Contiguous)); every other window is walked through the layout.
fn storage_of(geometry: &Geometry, axes: &[Axis], labelled: bool, launch: &Launcher) -> Storage {
    if !(labelled && geometry.tiling().is_tiled()) {
        return Storage::Strided;
    }
    let labels = geometry.labels(axes);
    let tiles = StoragePartitioning::new(geometry, &labels)
        .map(|storage| storage.contiguous_tiles(&geometry.extents(&labels)))
        .unwrap_or_default();
    let (space, levels) = (launch.space(), launch.partitioning().levels());
    let cuts_to = |level: usize, tile: &[(Axis, usize)]| {
        let leaf = space.leaf(&levels[..=level]);
        axes.iter().all(|&axis| {
            let stored = tile
                .iter()
                .find(|&&(a, _)| a == axis)
                .map_or(1, |&(_, e)| e);
            leaf.extent(axis) == stored
        })
    };
    let level = tiles
        .iter()
        .rev()
        .find_map(|tile| (0..levels.len()).find(|&level| cuts_to(level, tile)));
    Storage::Tiled(level)
}

/// The binding as the arg ships it, its tiling restated over the settled geometry's dims.
fn settled_tensor(mut binding: TensorBinding, geometry: &Geometry) -> TensorArg {
    binding.shape = geometry.shape().into();
    binding.strides = geometry.strides().into();
    binding.tiling = geometry.tiling();
    binding.into_tensor_arg()
}

/// A bound operand: its tensor argument, comptime [`TileSpec`] and served width.
pub struct Bound {
    tensor: Option<TensorArg>,
    /// Served width (values per line); a packed binding is narrower by the packing factor.
    pub vector_size: usize,
    pub spec: TileSpec,
}

impl Bound {
    /// The operand as the kernel's [`TileArg`](crate::TileArg) launch argument.
    pub fn arg<E: Numeric, V: Size>(self) -> TileArgLaunch<'static, E, V> {
        let Bound { spec, .. } = &self;
        let spec = spec.clone();
        TileArgLaunch::new(self.tensor(), spec)
    }

    /// The tensor argument itself; panics for an operand built over geometry alone.
    pub fn tensor(self) -> TensorArg {
        self.tensor
            .expect("Bound: this operand was built over geometry alone and has no tensor to bind")
    }

    /// The width the binding is typed at, narrower than `vector_size` when packed.
    pub fn bound_width(&self) -> usize {
        self.spec.packing.physical(self.vector_size)
    }
}

/// What [`build_spec`](Arg::build_spec) settles for an operand with no tensor to bind.
pub struct Unbound {
    pub spec: TileSpec,
    /// Served width (values per line).
    pub vector_size: usize,
    pub geometry: Geometry,
}
