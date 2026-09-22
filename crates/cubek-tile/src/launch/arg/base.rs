//! The [`Arg`] builder: the one place a launched tensor becomes an operand. Every client binds
//! its operands through it, so the layout, broadcast and bounds-check derivation lives here and
//! nowhere else.

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::zspace::Tiling;

use super::analysis::{Boundaries, Labels, Refusal, StorageLevel};
use crate::{
    Axis, Boundary, Field, Geometry, Launcher, Packing, Projection, QuantTileArgLaunch,
    Quantization, Storage, StorageTiling, TileArgLaunch, TileSpec,
};

/// Typestate marker: the operand's axes are not yet stated.
pub struct Unlabelled;
/// Typestate marker: the operand's axes are stated, by label or by a gathering projection.
pub struct Labelled;

/// Whether an operand's reads are bounds-checked, and how.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum BoundaryPolicy {
    /// Checked with [`Boundary::Zero`] on the axes that can leave the buffer: the ones the
    /// partitioning overhangs, or every axis of a mapping that may underflow.
    #[default]
    Derived,
    /// Unchecked on every axis: the launch proves every read in bounds.
    Unchecked,
    /// Checked with this boundary on every axis that is not provably in bounds.
    Every(Boundary),
}

/// What the builder accumulates; the typestate lives in the wrapper, not here.
struct ArgData<'a> {
    launch: &'a Launcher,
    /// The tensor this operand is served from, when there is one. `None` for a destination with
    /// no address: a fused store writes through a call, so the launch has nothing to bind and only
    /// the comptime half is derived ([`build_spec`](Arg::build_spec)).
    binding: Option<TensorBinding>,
    /// The operand's physical extents and strides: the source of truth for the whole derivation.
    /// A bound operand copies them off its binding, an unbound one states them.
    geometry: Geometry,
    axes: &'a [Axis],
    batches: &'a [Axis],
    /// The operand's own affine mapping, when it states one ([`gathered`](Arg::gathered)); `None`
    /// derives it from the labelled dims.
    projection: Option<Projection>,
    width: usize,
    boundary: BoundaryPolicy,
    packing: Packing,
    quant: Option<Quantization>,
    /// Whether the labelled dims bind in the order they step rather than the order the binding
    /// names them ([`in_stride_order`](Arg::in_stride_order)).
    in_stride_order: bool,
}

/// One operand of a launch, being described: which axes its buffer spans, how wide it is served,
/// how its edges are checked. Built from [`Launcher::arg`] or [`Launcher::unbound`]; `S` says
/// whether the axes are stated yet, which is what makes [`build`](Self::build) exist.
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
                quant: None,
                in_stride_order: false,
            },
            _state: PhantomData,
        }
    }

    /// The axes the operand's trailing buffer dims carry, `[row, col]` for a matmul operand; the
    /// leading dims are its [`batches`](Arg::batches). Exclusive with [`gathered`](Self::gathered).
    pub fn axes(mut self, axes: &'a [Axis]) -> Arg<'a, Labelled> {
        self.data.axes = axes;
        self.labelled()
    }

    /// An explicit affine [`Projection`] for a gathered operand (a convolution's input, a
    /// resample's source), from logical axes to buffer dims. Refuses a storage-tiled binding.
    ///
    /// Checking follows [`may_underflow`](Projection::may_underflow); a window off the buffer's
    /// *tail* is not detected, so a gather that overruns (a rational mapping's last window always
    /// does) must state [`BoundaryPolicy::Every`].
    ///
    /// An axis sharing a physical dim has no extent here, so a [`Dynamic`](crate::Extent) one needs
    /// another operand to witness it; dynamic scales, divisors and offsets declare a launch bound.
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
    /// The outer (batch) axes in the output's order, right-aligned to this operand's leading
    /// dims (numpy broadcast): pass the full list, extra leading axes are the ones this operand
    /// omits, and a size-1 dim drops out. Default none (unbatched).
    pub fn batches(mut self, axes: &'a [Axis]) -> Self {
        self.data.batches = axes;
        self
    }

    /// Bind the labelled dims in the order they step, coarsest first: a transposed view binds as
    /// the buffer it is (same bytes, no copy), so its unit-strided dim is innermost, serving lines.
    /// Labels follow dims: `axes(&[K, N])` over `[n, k]` viewed `[k, n]` binds `[N, K]`.
    ///
    /// Batch dims stay where they are: a broadcast one strides by zero, and where it sorts says
    /// nothing about which way the operand's own dims run.
    pub fn in_stride_order(mut self) -> Self {
        self.data.in_stride_order = true;
        self
    }

    /// Serve the innermost axis in `width`-wide lines (default `1`, scalar). Only valid when that
    /// axis is contiguous. The kernel's element type carries the width (`Vector<E, V>`).
    pub fn vectorize(mut self, width: usize) -> Self {
        self.data.width = width;
        self
    }

    /// How this operand's edges are checked. Whatever the policy, the check lands only on the axes
    /// that can leave the buffer, which is what keeps a vectorized innermost axis unmasked.
    pub fn boundary(mut self, policy: BoundaryPolicy) -> Self {
        self.data.boundary = policy;
        self
    }

    /// This operand's values are fields of a stored word, `field` wide each. A fact of the values
    /// alone: the binding's shape and strides count *values*, and this says how many share a word.
    /// Scales are a second tensor and a second operand; nothing here decodes behind a read.
    pub fn packed(mut self, field: impl Into<Field>) -> Self {
        self.data.packing = Packing::Packed {
            field: field.into(),
        };
        self
    }

    /// The operand is quantized: its binding holds the scheme's storage element (declared in
    /// values), and `quant` says how reads dequantize. [`build`](Self::build)'s product carries
    /// it and launches as a [`QuantTileArg`](crate::QuantTileArg).
    pub fn quantized(mut self, quant: Quantization) -> Self {
        self.data.quant = Some(quant);
        self
    }

    /// The operand, bound: its tensor argument, its comptime [`TileSpec`] and the served width.
    ///
    /// # Panics
    ///
    /// With the [`Refusal`] that names what this operand cannot be: a storage tile no level cuts
    /// to, a width its buffer does not serve, a vector line that is not provably in bounds.
    pub fn build(self) -> Bound {
        let (bound, _) = self.realize().unwrap_or_else(|refusal| panic!("{refusal}"));
        bound
    }

    /// The untensored half: everything [`build`](Self::build) would derive but the tensor argument
    /// itself, for a destination with no address (a fused store writes through a call), with the
    /// geometry the derivation settled on.
    ///
    /// # Panics
    ///
    /// As [`build`](Self::build).
    pub fn build_spec(self) -> Unbound {
        let (bound, geometry) = self.realize().unwrap_or_else(|refusal| panic!("{refusal}"));
        Unbound {
            spec: bound.spec,
            vector_size: bound.vector_size,
            geometry,
        }
    }

    /// The derivation both builds share: settle the operand's [`Projection`] (the labelled dims,
    /// or the gathered mapping as given), which level its storage tile is the tile of, where the
    /// bounds-check lands, and whether the buffer serves the width; then mint the [`TileSpec`].
    fn realize(self) -> Result<(Bound, Geometry), Refusal> {
        let data = self.data;
        let launch = data.launch;
        let width = data.width;
        // How the bound tensor says it is stored: the tiling is a fact of the tensor, read off its
        // binding rather than stated at the launch. An unbound operand (a fused store) has none to
        // ask; a gathered operand states its mapping and refuses a tiled binding.
        let stored = data.binding.as_ref().map(|b| b.tiling).unwrap_or_default();
        let stated = data.projection.is_some() || stored.is_tiled();
        let (geometry, axes) =
            stride_ordered(data.geometry, data.axes, data.in_stride_order, stated)?;
        let (tiling, storage) =
            storage_of(&geometry, &axes, data.projection.is_none(), stored, launch)?;
        let labels = match data.projection {
            Some(projection) => {
                Labels::stated(&geometry, launch, projection, &axes, data.batches, stored)?
            }
            None => Labels::new(&geometry, &axes, data.batches, tiling.clone())?,
        };
        let Labels {
            geometry,
            projection,
            addressed,
        } = labels;
        projection.validate(width);
        // The width against the *settled* geometry, the one the kernel re-expresses in lines.
        // `Launcher::vector_size` derives a width that divides; a stated one (pinned, or a fused
        // destination the negotiation never saw) is gated here: `stride / width` truncates silently.
        geometry
            .serves_lines(width)
            .map_err(|why| Refusal::WidthNotServed { width, why })?;
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
        if let Some(quant) = &data.quant {
            quant.check(&spec, launch.space(), width)?;
        }
        let tensor = data
            .binding
            .map(|binding| settled_tensor(binding, &geometry, tiling.as_ref(), stored));
        let bound = Bound {
            tensor,
            vector_size: width,
            spec,
            quant: data.quant,
        };
        Ok((bound, geometry))
    }
}

/// The labelled dims in the order the buffer steps them where the caller asked for it, settled
/// before the tiling and the labelling read them, so both see one order. A stated layout (a
/// gathered mapping, a storage-tiled binding) already says how it steps and is refused.
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

/// The operand's storage tiling, read off its binding, and the level whose tile it is: a storage
/// tile is the tile of one of the kernel's levels, or the operand is refused. A gathered operand
/// (`labelled == false`) states its own mapping and reads no tiling.
fn storage_of(
    geometry: &Geometry,
    axes: &[Axis],
    labelled: bool,
    stored: Tiling,
    launch: &Launcher,
) -> Result<(Option<StorageTiling>, Storage), Refusal> {
    if !(labelled && stored.is_tiled()) {
        return Ok((None, Storage::Strided));
    }
    let tiling = StorageTiling::stored(stored, axes.len(), geometry.rank());
    let level = StorageLevel::new(
        geometry,
        axes,
        &tiling,
        launch.space(),
        launch.partitioning().levels(),
    )?;
    Ok((Some(tiling), level.storage()))
}

/// The binding as the arg ships it: the settled geometry, whose derivation may have dropped
/// broadcast batch dims, and its tiling restated over those dims. A `Tiling` counts fragments off
/// the leading dims, so one counted off a dropped dim claims a layout the arg lacks.
fn settled_tensor(
    mut binding: TensorBinding,
    geometry: &Geometry,
    tiling: Option<&StorageTiling>,
    stored: Tiling,
) -> TensorArg {
    binding.shape = geometry.shape().into();
    binding.strides = geometry.strides().into();
    binding.tiling = match tiling {
        Some(tiling) => {
            let batch_dims = geometry.rank() - tiling.physical_rank();
            let mut fragments = vec![1; batch_dims];
            fragments.extend((0..tiling.rank()).map(|axis| tiling.fragments(axis)));
            Tiling::new(&fragments)
                .expect("the binding's own tiling fit, and this drops dims from it")
        }
        None => stored,
    };
    binding.into_tensor_arg()
}

/// A bound operand: its tensor argument (absent for an operand built over geometry alone), its
/// comptime [`TileSpec`], the served width, and its quantization when it has one.
pub struct Bound {
    tensor: Option<TensorArg>,
    /// Served width (values per line); a packed binding is narrower by the packing factor.
    pub vector_size: usize,
    pub spec: TileSpec,
    pub quant: Option<Quantization>,
}

impl Bound {
    /// The operand as the kernel's [`TileArg`](crate::TileArg) launch argument.
    pub fn arg<E: Numeric, V: Size>(self) -> TileArgLaunch<'static, E, V> {
        let Bound { spec, .. } = &self;
        let spec = spec.clone();
        TileArgLaunch::new(self.tensor(), spec)
    }

    /// The operand as the kernel's [`QuantTileArg`](crate::QuantTileArg) launch argument: values,
    /// scales, spec and scheme as one thing. Read [`bound_width`](Self::bound_width) first.
    ///
    /// # Panics
    ///
    /// When the operand was not [`quantized`](Arg::quantized).
    pub fn quant_arg<E: Numeric, V: Size>(self) -> QuantTileArgLaunch<'static, E, V> {
        let quant = self
            .quant
            .expect("Bound::quant_arg: this operand was not quantized");
        quant.arg(
            self.tensor.expect("a quantized operand is always bound"),
            self.spec,
        )
    }

    /// The tensor argument itself, for a launch that binds it under another argument type.
    pub fn tensor(self) -> TensorArg {
        self.tensor
            .expect("Bound: this operand was built over geometry alone and has no tensor to bind")
    }

    /// The width the binding is typed at: the launch value for the kernel's `Size` generic, while
    /// [`vector_size`](Self::vector_size) is what the operand *serves*, held by a packed store in
    /// fewer words. The two agree without packing.
    pub fn bound_width(&self) -> usize {
        match &self.quant {
            Some(quant) => self.vector_size / quant.num_quants(),
            None => self.spec.packing.physical(self.vector_size),
        }
    }
}

/// What [`build_spec`](Arg::build_spec) settles for an operand with no tensor to bind: the
/// comptime [`TileSpec`], the served width, and the geometry a bound `TensorArg` would ship,
/// broadcast dims dropped; [`Tile::of_sink`](crate::Tile::of_sink) addresses through it.
pub struct Unbound {
    pub spec: TileSpec,
    /// Served width (values per line).
    pub vector_size: usize,
    pub geometry: Geometry,
}
