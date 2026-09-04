//! The [`StridedTileSource`] builder: the one place a launched tensor becomes a
//! [`StridedOperand`]. Every client (matmul, dequantize, …) loads tiles through it, so
//! the layout/broadcast wiring lives here, not at each call site.

use core::marker::PhantomData;

use cubecl::prelude::*;

use cubecl::quant::scheme::{QuantScheme, QuantValue};
use cubecl::std::tensor::layout::linear::linear_view;

use crate::{
    Axis, Boundary, ConcreteLayout, DequantAt, Geometry, Level, Nest, Packing, PhysicalAxis,
    Projection, QuantTileArgLaunch, Space, StorageTiling, TileArgLaunch, TileSpec, validate_scheme,
};

/// Typestate marker: a required [`StridedTileSource`] field has been set.
pub struct Set;
/// Typestate marker: a required [`StridedTileSource`] field is still missing.
pub struct Unset;

/// The fields an [`StridedTileSource`] accumulates; the typestate lives in the wrapper, not here.
struct TileSourceData<'a> {
    /// The tensor this operand is served from, when there is one. `None` for a
    /// destination that has no address: a fused store writes through a call, so
    /// the launch has nothing to bind, and only the comptime half is derived
    /// ([`build_spec`](StridedTileSource::build_spec)).
    binding: Option<TensorBinding>,
    /// The operand's physical extents and strides. The source of truth for the
    /// whole derivation: a bound operand copies them off its binding, an
    /// unbound one states them, and both reach [`labeled`] the same way.
    geometry: Geometry,
    space: Option<&'a Space>,
    /// The concrete (real-extent) space and its levels, when minted by a
    /// [`Launcher`](crate::Launcher): lets [`build`](StridedTileSource::build) derive the
    /// bounds-check from overhang.
    concrete: Option<&'a Nest>,
    subspace: &'a [Axis],
    batch_axes: &'a [Axis],
    /// How the subspace axes are storage-tiled in the binding; `None` is untiled.
    tiling: Option<StorageTiling>,
    /// The operand's own affine mapping, when it states one ([`gathered`](StridedTileSource::gathered));
    /// `None` derives it from the labeled dims instead.
    projection: Option<Projection>,
    v: usize,
    boundary: Option<Option<Boundary>>,
    /// How this operand's values sit in its binding: several to a stored word, or as they are.
    packing: Packing,
    /// The launch's cube size (units per cube); set by [`Launcher::arg`](crate::Launcher::arg).
    units: usize,
    /// Present when the operand is quantized; [`realize`](StridedTileSource::realize) validates it.
    quant: Option<Quantization>,
}

/// Typestate builder for a strided tile kernel operand, started with
/// [`Launcher::arg`](crate::Launcher::arg) or [`StridedOperand::source`]. `Sp`/`Sub` make
/// [`build`](Self::build) exist only once both required setters are [`Set`]; `Q` records whether
/// [`quantized`](Self::quantized) was called, so `build` returns a [`StridedOperand`] or a
/// [`QuantOperand`] and no call site ever probes an option.
pub struct StridedTileSource<'a, Sp, Sub, Q> {
    data: TileSourceData<'a>,
    _state: PhantomData<(Sp, Sub, Q)>,
}

impl<'a> StridedTileSource<'a, Unset, Unset, Unset> {
    pub(crate) fn new(binding: TensorBinding) -> Self {
        let geometry = Geometry::from(&binding);
        Self::over(Some(binding), geometry)
    }

    /// The same builder over geometry alone, for an operand with no tensor to
    /// bind. See [`TileSourceData::binding`].
    pub(crate) fn of_geometry(geometry: &Geometry) -> Self {
        Self::over(None, geometry.clone())
    }

    fn over(binding: Option<TensorBinding>, geometry: Geometry) -> Self {
        StridedTileSource {
            data: TileSourceData {
                binding,
                geometry,
                space: None,
                concrete: None,
                subspace: &[],
                batch_axes: &[],
                tiling: None,
                projection: None,
                v: 1,
                boundary: None,
                packing: Packing::Plain,
                units: 0,
                quant: None,
            },
            _state: PhantomData,
        }
    }
}

impl<'a, Sp, Sub, Q> StridedTileSource<'a, Sp, Sub, Q> {
    /// The global iteration space this operand projects from (required).
    pub fn space(mut self, space: &'a Space) -> StridedTileSource<'a, Set, Sub, Q> {
        self.data.space = Some(space);
        StridedTileSource {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// The inner block of axes the operand iterates, its `[row, col]` for a matmul (required
    /// unless [`gathered`](Self::gathered) states the mapping instead, non-empty). Complementary
    /// to [`batches`](Self::batches), the outer dims.
    pub fn subspace(mut self, axes: &'a [Axis]) -> StridedTileSource<'a, Sp, Set, Q> {
        self.data.subspace = axes;
        StridedTileSource {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// Sets an explicit affine [`Projection`] for a gathered operand (convolution, resample),
    /// mapping logical axes to buffer dimensions. Mutually exclusive with
    /// [`subspace`](Self::subspace), [`batches`](Self::batches) and [`tiling`](Self::tiling).
    ///
    /// Checking follows [`may_underflow`](Projection::may_underflow); a window off the buffer's
    /// *tail* is not detected, so a gather that overruns (a rational mapping's last window always
    /// does) must state [`checked(true)`](Self::checked). An axis sharing a physical dim has no
    /// extent here, so a [`Dynamic`](crate::Extent) one needs another operand to witness it.
    /// Dynamic scales, divisors and offsets declare a bound the launch must stay within.
    pub fn gathered(mut self, projection: Projection) -> StridedTileSource<'a, Sp, Set, Q> {
        self.data.projection = Some(projection);
        StridedTileSource {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// The outer (batch) axes in the output's order, right-aligned to this operand's leading
    /// dims (numpy broadcast): pass the full list, extra leading axes are the ones this operand
    /// omits, and a size-1 dim drops out. Default none (unbatched).
    pub fn batches(mut self, axes: &'a [Axis]) -> Self {
        self.data.batch_axes = axes;
        self
    }

    /// How this binding storage-tiles the [`subspace`](Self::subspace) axes: one fragment count
    /// per subspace axis, laid out level-major behind the batch dims. Default untiled (one
    /// physical dim per axis). Only labels the dims, so the tiling is read back off the
    /// [`ConcreteLayout`](crate::ConcreteLayout) rather than declared twice.
    pub fn tiling(mut self, tiling: StorageTiling) -> Self {
        self.data.tiling = Some(tiling);
        self
    }

    /// Serve the innermost axis in `v`-wide lines (default `1`, i.e. scalar). Only valid when
    /// that axis is contiguous. The kernel's element type carries the width (`Vector<E, V>`).
    pub fn vectorize(mut self, v: usize) -> Self {
        self.data.v = v;
        self
    }

    /// Force the overhang bounds-check on or off, using [`Boundary::Zero`] when checked. Default:
    /// derived from the concrete space by the [`Launcher`](crate::Launcher). Overwrites whatever
    /// mode stood, so sequence a `Clamp` [`with_boundary`](Self::with_boundary) *after* this, not
    /// before. States the mode, not the axis list: [`build`](Self::build) still lands it only on
    /// the axes that can leave the buffer, which is what keeps a vectorized innermost axis.
    pub fn checked(mut self, check: bool) -> Self {
        self.data.boundary = Some(if check { Some(Boundary::Zero) } else { None });
        self
    }

    /// Set the boundary handling mode for out-of-bounds access explicitly (`None` forces
    /// unchecked, `Some` forces checked at that mode). Like [`checked`](Self::checked), a `Some`
    /// states the mode and [`build`](Self::build) picks the axes it lands on.
    pub fn with_boundary(mut self, boundary: Option<Boundary>) -> Self {
        self.data.boundary = Some(boundary);
        self
    }

    /// This operand's values are fields of a stored word, `field` wide each. A fact of the values
    /// alone: the binding's shape and strides count *values*, and this says how many share a word.
    /// Scales are a second tensor and a second operand; nothing here decodes behind a read.
    pub fn packed(mut self, field: QuantValue) -> Self {
        self.data.packing = Packing::Packed { field };
        self
    }

    /// The concrete (real-extent) space the bounds-check derives from; set by
    /// [`Launcher::arg`](crate::Launcher::arg).
    pub(crate) fn concrete(mut self, concrete: &'a Nest) -> Self {
        self.data.concrete = Some(concrete);
        self
    }

    /// The launch's cube size (units per cube); set by [`Launcher::arg`](crate::Launcher::arg).
    pub(crate) fn cube_units(mut self, units: usize) -> Self {
        self.data.units = units;
        self
    }
}

impl<'a, Sp, Sub> StridedTileSource<'a, Sp, Sub, Unset> {
    /// Mark the operand as quantized: its binding holds the scheme's storage element (declared
    /// **in values**), and `scales` + `scheme` let reads dequantize into the served type.
    /// `scales` holds one binding per scheme level, innermost first. `dequant_at` says how far the
    /// quantized form travels, and rides here because that form ends at exactly one boundary, so
    /// one call says it once. Flips the typestate: `build` now yields a [`QuantOperand`].
    pub fn quantized(
        mut self,
        scales: &[TensorBinding],
        scheme: QuantScheme,
        dequant_at: DequantAt,
    ) -> StridedTileSource<'a, Sp, Sub, Set> {
        self.data.quant = Some(Quantization::new(scales, scheme, dequant_at));
        StridedTileSource {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// [`quantized`](Self::quantized) for a lookup scheme
    /// ([`QuantMode::Lookup`](cubecl::quant::scheme::QuantMode)): each stored field indexes
    /// `table` and a read reconstructs `table[field] * scale`. The table must hold `2^bits` f32
    /// entries, unchecked here: the unpack's mask bounds every index to that range.
    /// Public for the same reason [`quantized`](Self::quantized) is: it is the only way to
    /// declare a lookup operand, since the table has no other binding point.
    pub fn quantized_lookup(
        mut self,
        scales: TensorArg,
        table: BufferArg,
        scheme: QuantScheme,
        dequant_at: DequantAt,
    ) -> StridedTileSource<'a, Sp, Sub, Set> {
        self.data.quant = Some(Quantization::lookup(scales, table, scheme, dequant_at));
        StridedTileSource {
            data: self.data,
            _state: PhantomData,
        }
    }
}

/// How an operand is quantized: the scales beside its values, the scheme saying how to fold them
/// back in, and how far the quantized form travels before something decodes it. One thing, because
/// none of the three says anything on its own: a scheme without scales cannot be applied, and an
/// [`DequantAt`] without a scheme has nothing to bound.
pub struct Quantization {
    /// The innermost level's scales, the only ones addressed per position.
    pub scales: TensorArg,
    /// The global level's scale, one for the whole tensor, present exactly when the scheme has a
    /// second level ([`validate`](Self::validate) holds the two together).
    pub global: Option<TensorBinding>,
    /// A lookup scheme's `2^bits`-entry table, present exactly under
    /// [`QuantMode::Lookup`](cubecl::quant::scheme::QuantMode);
    /// [`validate`](Self::validate) holds the two together.
    pub table: Option<BufferArg>,
    pub scheme: QuantScheme,
    pub dequant_at: DequantAt,
}

impl Quantization {
    /// `scales` holds one binding per scheme level, innermost first. Only the innermost level is
    /// addressed per position; a global level is read once from its first element. Checks only
    /// that the slice holds 1 or 2 bindings; matching `scheme`'s level count is
    /// [`validate`](Self::validate)'s job.
    pub fn new(scales: &[TensorBinding], scheme: QuantScheme, dequant_at: DequantAt) -> Self {
        let (inner, global) = match scales {
            [inner] => (inner, None),
            [inner, global] => (inner, Some(global.clone())),
            _ => panic!(
                "StridedTileSource::quantized: {} scale bindings, expected 1 or 2 (innermost first)",
                scales.len()
            ),
        };
        Quantization {
            scales: inner.clone().into_tensor_arg(),
            global,
            table: None,
            scheme,
            dequant_at,
        }
    }

    /// [`new`](Self::new) with a lookup scheme's table beside the scales.
    pub fn lookup(
        scales: TensorArg,
        table: BufferArg,
        scheme: QuantScheme,
        dequant_at: DequantAt,
    ) -> Self {
        Quantization {
            scales,
            global: None,
            table: Some(table),
            scheme,
            dequant_at,
        }
    }

    /// Values per stored element: `1` unless the scheme packs several into each.
    pub fn num_quants(&self) -> usize {
        self.scheme.num_quants()
    }

    /// Refuse what this quantization cannot serve, on the caller's thread: the scheme against the
    /// operand's cuts and served width. Where the [`DequantAt`] can be honoured is the fragment
    /// load's to say, at the kernel's own call.
    pub(crate) fn validate(&self, space: &Space, levels: Option<&[Level]>, vector_size: usize) {
        cubecl::std::quant::check_scale_bindings(&self.scheme, 1 + self.global.is_some() as usize);
        validate_scheme(space, levels.unwrap_or(&[]), vector_size, self.scheme);
        cubecl::std::quant::check_table_bindings(&self.scheme, self.table.is_some());
    }
}

/// A built plain operand: the tensor argument, its comptime [`TileSpec`], and the served
/// width (also the binding width: the launch value for its `Size` generic).
pub struct StridedOperand {
    pub tensor: TensorArg,
    pub vector_size: usize,
    pub spec: TileSpec,
}

impl StridedOperand {
    /// The operand as the kernel's [`TileArg`](crate::TileArg) launch argument.
    pub fn arg<E: Numeric, V: Size>(self) -> TileArgLaunch<'static, E, V> {
        TileArgLaunch::new(self.tensor, self.spec)
    }
}

/// What [`build_spec`](StridedTileSource::build_spec) settles for an operand with no tensor to
/// bind: the comptime [`TileSpec`], the served width, and the derived geometry. That geometry is
/// what a bound operand's `TensorArg` would have shipped, not what the caller stated (labeling
/// drops broadcast batch dims), and [`Tile::of_sink`](crate::Tile::of_sink) addresses through it.
pub struct DerivedSpec {
    pub spec: TileSpec,
    /// Served width (values per line).
    pub vector_size: usize,
    pub geometry: Geometry,
}

/// A built quantized operand: the storage-typed tensor and its spec, plus the scales and
/// the comptime scheme as first-class fields; nothing to probe or drain at the call site.
pub struct QuantOperand {
    pub tensor: TensorArg,
    /// Served width (values per line); the binding is narrower by the packing factor.
    pub vector_size: usize,
    pub spec: TileSpec,
    pub quant: Quantization,
}

impl QuantOperand {
    /// The width the binding is typed at: the launch value for the kernel's `Size`
    /// generic. A packed store's buffer is narrower than the served width by the packing
    /// factor ([`tile_dequant`](crate::TileArg::tile_dequant) serves binding width × pack).
    pub fn bound_width(&self) -> usize {
        self.vector_size / self.quant.num_quants()
    }

    /// The operand as the kernel's [`QuantTileArg`](crate::QuantTileArg) launch argument:
    /// values, scales, spec and scheme as one thing. Read
    /// [`bound_width`](Self::bound_width) before consuming.
    pub fn arg<E: Numeric, V: Size>(self) -> QuantTileArgLaunch<'static, E, V> {
        QuantTileArgLaunch::new(
            self.tensor,
            self.quant.scales,
            self.quant.global.map(linear_view).into(),
            self.quant.table.into(),
            self.spec,
            self.quant.scheme,
            self.quant.dequant_at,
        )
    }
}

impl StridedOperand {
    /// Start describing a strided tile kernel operand sourced from `binding`. Set the required
    /// [`space`](StridedTileSource::space) and either [`subspace`](StridedTileSource::subspace) or
    /// [`gathered`](StridedTileSource::gathered); `build` will not compile until both are set.
    /// Residency defaults to reading in place, so an operand a fragment load cannot address must
    /// state where it is materialized.
    pub fn source<'a>(binding: TensorBinding) -> StridedTileSource<'a, Unset, Unset, Unset> {
        StridedTileSource::new(binding)
    }
}

/// [`realize`](StridedTileSource::realize)'s product, consumed by the two typed builds.
struct Realized {
    /// `None` exactly when the source was built over geometry alone.
    tensor: Option<TensorArg>,
    vector_size: usize,
    spec: TileSpec,
    quant: Option<Quantization>,
    /// The geometry the derivation settled on, which is what `tensor` ships when
    /// there is one. See [`DerivedSpec`].
    geometry: Geometry,
}

impl<'a, Q> StridedTileSource<'a, Set, Set, Q> {
    /// The derivation both builds share: settle the operand's [`Projection`] (the labeled dims
    /// folded into a [`ConcreteLayout`], or the [`gathered`](StridedTileSource::gathered) mapping
    /// as given), derive the bounds-check, and mint the comptime [`TileSpec`].
    fn realize(self) -> Realized {
        let TileSourceData {
            binding,
            mut geometry,
            space,
            concrete,
            batch_axes,
            subspace,
            tiling,
            projection,
            v,
            boundary,
            packing,
            units,
            quant,
        } = self.data;
        let space = space.unwrap();

        // Use the explicit projection if gathered, or derive it from labeled axes.
        // `addressed` contains all logical axes used for bounds checking.
        let (projection, addressed) = match projection {
            Some(projection) => {
                check_stated(&geometry, space, &projection, subspace, batch_axes, &tiling);
                let addressed = projection.logical_axes().to_vec();
                (projection, addressed)
            }
            None => labeled(&mut geometry, subspace, batch_axes, tiling),
        };

        // Derive boundary check: use explicit override if set, otherwise check for overhang or underflow.
        let boundary = boundary.unwrap_or_else(|| match concrete {
            Some(concrete) => {
                let overhangs = addressed
                    .iter()
                    .filter(|&&axis| concrete.space.contains(axis))
                    .any(|&axis| concrete.space.overhangs(&concrete.levels, axis));
                (overhangs || projection.may_underflow()).then_some(Boundary::Zero)
            }
            None => Some(Boundary::Zero),
        });

        projection.validate(v);
        // The width against the *settled* geometry, which is the one the kernel re-expresses in
        // lines. `Launcher::vector_size` derives a width that divides; a caller that states one
        // (a pinned width, a fused destination the negotiation never saw) reaches here, and
        // the failure it prevents is silent: `stride / v` truncates in bounds and addresses a
        // fraction of the operand.
        if let Err(why) = geometry.serves_lines(v) {
            panic!("StridedTileSource::vectorize: this operand cannot be served {v} wide: {why}");
        }
        let coords = projection.untiled();
        let coord_rank = projection.coordinate_rank();
        assert!(
            coord_rank > 0,
            "StridedTileSource: an operand must address at least one coordinate"
        );

        // Whether coordinate axis `pa` is inside the buffer by construction. Only an identity map
        // can be: it reaches exactly as far as its own coordinate, so it stays inside whenever its
        // tiling divides. Its `Offset::Static(0)` is also what keeps it clear of the *underflow*
        // half of the derivation above: a negative or `Dynamic` offset makes a map non-identity,
        // so the axis whose origin can fall below zero is never one settled here. An affine map
        // reaches further than any axis extent describes, and how far is the caller's to size the
        // buffer for, which is the trust the derivation above already runs on; no proof here can
        // retire its policy.
        let settled = |pa: usize| match coords.physical_axis(pa).identity_axis() {
            // An axis the concrete space does not describe is unproven, not proven: the
            // derivation above already skips it when *arming* the mode, so nothing here may use
            // that same silence to drop one.
            Some(axis) => concrete.is_some_and(|concrete| {
                concrete.space.contains(axis) && !concrete.space.overhangs(&concrete.levels, axis)
            }),
            None => false,
        };

        // A vector line only needs a scalar fallback when its own innermost axis is unsettled.
        // Other coordinate axes may still be masked or clamped independently (NHWC interpolation
        // clamps H/W while serving a contiguous C line).
        let innermost_unsettled = !settled(coord_rank - 1);
        assert!(
            !(boundary.is_some() && v > 1 && innermost_unsettled),
            "StridedTileSource: the innermost axis is served in vector lines but is not provably \
             in bounds (it overhangs its tiling, or its map is affine and reaches past any extent \
             stated here); serve it scalar, or state `checked(false)` if the launch proves its \
             vector lines are in bounds"
        );

        // The mode covers the coordinate axes that can leave the buffer, and only those: a settled
        // axis would pay for a mask that can never fire, and a settled *innermost* axis must be
        // left alone outright, since a window clamps in lines and would alias the edge line.
        let boundaries = (0..coord_rank)
            .map(|pa| boundary.filter(|_| !settled(pa)))
            .collect::<Vec<_>>();

        let spec = TileSpec::new(projection)
            .boundaries(&boundaries)
            .units(units)
            .packing(packing);
        if let Some(quant) = &quant {
            // Quantization is not supported for gathered operands.
            assert!(
                spec.projection.untiled().is_direct(),
                "StridedTileSource::quantized: a gathered operand cannot be quantized; its scale \
                 grid is shaped over its logical axes, which its buffer's dims no longer match"
            );
            quant.validate(
                &space.project(spec.axes()),
                concrete.map(|c| c.levels.as_slice()),
                v,
            );
        }
        Realized {
            tensor: binding.map(|mut binding| {
                // The derivation may have dropped broadcast batch dims; the arg
                // ships the geometry it settled on, not the one it arrived with.
                binding.shape = geometry.shape().into();
                binding.strides = geometry.strides().into();
                binding.into_tensor_arg()
            }),
            vector_size: v,
            spec,
            quant,
            geometry,
        }
    }
}

/// Derives a [`Projection`] from labeled subspace and batch axes. Leading batch dims align with
/// `batch_axes` (size-1 broadcast dims omitted), the inner subspace axes follow `tiling`. Returns
/// the projection plus every physical axis prior to broadcast omission, for bounds checking.
fn labeled(
    geometry: &mut Geometry,
    subspace: &[Axis],
    batch_axes: &[Axis],
    tiling: Option<StorageTiling>,
) -> (Projection, Vec<Axis>) {
    let rank = geometry.rank();
    let tiling = tiling.unwrap_or_else(|| StorageTiling::uniform(subspace.len(), 0));
    assert_eq!(
        tiling.rank(),
        subspace.len(),
        "StridedTileSource: the tiling describes {} axes but the subspace has {}",
        tiling.rank(),
        subspace.len()
    );
    let block = tiling.order(subspace);
    let block_dims = block.len();
    assert!(
        rank >= block_dims,
        "StridedTileSource: operand rank {rank} is smaller than its subspace block of {block_dims} dims ({} axes over {tiling:?})",
        subspace.len()
    );
    let batch_dims = rank - block_dims;
    assert!(
        batch_dims <= batch_axes.len(),
        "StridedTileSource: {batch_dims} batch dims but only {} batch axes given",
        batch_axes.len()
    );
    let mut physical_axes = Vec::with_capacity(rank);
    physical_axes.extend_from_slice(&batch_axes[batch_axes.len() - batch_dims..]);
    physical_axes.extend_from_slice(&block);

    let mut phys = Vec::new();
    let mut dims = Vec::new();

    for (&axis, (extent, stride)) in physical_axes.iter().zip(geometry.dims()) {
        // A subspace axis never drops out, however small, since the tile is shaped over it.
        if batch_axes.contains(&axis) && extent == 1 && !subspace.contains(&axis) {
            continue;
        }
        phys.push(PhysicalAxis::new(axis, extent));
        dims.push((extent, stride));
    }

    *geometry = Geometry::of_dims(&dims);
    (
        Projection::of_layout(&ConcreteLayout::new(&phys)),
        physical_axes,
    )
}

/// Validates an explicit gathered [`Projection`] against the tensor binding and iteration space.
/// Ensures mutually exclusive labeling options (`subspace`, `batch_axes`, `tiling`) were not provided.
fn check_stated(
    geometry: &Geometry,
    space: &Space,
    projection: &Projection,
    subspace: &[Axis],
    batch_axes: &[Axis],
    tiling: &Option<StorageTiling>,
) {
    assert!(
        subspace.is_empty() && batch_axes.is_empty() && tiling.is_none(),
        "StridedTileSource::gathered: the mapping is stated outright, so `subspace`, `batches` \
         and `tiling` have nothing left to describe"
    );
    assert_eq!(
        projection.physical_rank(),
        geometry.rank(),
        "StridedTileSource::gathered: the mapping addresses {} dims but the operand has {}",
        projection.physical_rank(),
        geometry.rank()
    );
    for &axis in projection.logical_axes() {
        assert!(
            space.contains(axis),
            "StridedTileSource::gathered: the mapping spans {axis:?}, which the launched space \
             does not have"
        );
    }
}

impl<'a> StridedTileSource<'a, Set, Set, Unset> {
    /// Build the plain operand; the operand ships as a plain `TensorArg` plus its
    /// comptime [`TileSpec`].
    pub fn build(self) -> StridedOperand {
        let Realized {
            tensor,
            vector_size,
            spec,
            ..
        } = self.realize();
        StridedOperand {
            tensor: tensor.expect(
                "StridedTileSource::build: this source was built over geometry alone and has no \
                 tensor to bind; use `build_spec`",
            ),
            vector_size,
            spec,
        }
    }

    /// The untensored half: everything [`build`](Self::build) would derive but the tensor argument
    /// itself, for a destination with no address (a fused store writes through a call). The tile
    /// that walks it is projected, checked and staged exactly as a bound one is, so this is that
    /// same derivation rather than a restatement a caller would drift from. See [`DerivedSpec`].
    pub fn build_spec(self) -> DerivedSpec {
        let Realized {
            vector_size,
            spec,
            geometry,
            ..
        } = self.realize();
        DerivedSpec {
            spec,
            vector_size,
            geometry,
        }
    }
}

impl<'a, Q> StridedTileSource<'a, Set, Set, Q> {
    /// Build the quantized operand: the plain derivation plus its validated [`Quantization`].
    fn build_quant(self) -> QuantOperand {
        let Realized {
            tensor,
            vector_size,
            spec,
            quant,
            ..
        } = self.realize();
        QuantOperand {
            tensor: tensor.expect(
                "StridedTileSource::build_quant: a quantized operand is always bound; only a \
                 fused store is built over geometry alone",
            ),
            vector_size,
            spec,
            quant: quant.unwrap(),
        }
    }
}

impl<'a> StridedTileSource<'a, Set, Set, Set> {
    /// Build the quantized operand.
    pub fn build(self) -> QuantOperand {
        self.build_quant()
    }
}
