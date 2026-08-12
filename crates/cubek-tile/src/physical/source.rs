//! The [`StridedTileSource`] builder: the one place a launched tensor becomes a
//! [`StridedOperand`]. Every client (matmul, dequantize, …) loads tiles through it, so
//! the layout/broadcast wiring lives here, not at each call site.

use core::marker::PhantomData;

use cubecl::prelude::*;

use cubecl::quant::scheme::QuantScheme;

use crate::{
    Axis, Boundary, ConcreteLayout, DequantAt, Leaf, LoadMethod, PhysicalAxis, Projection,
    QuantTileArgLaunch, Space, StageStorage, StorageTiling, TileArgLaunch, TileSpec,
    validate_scheme,
};

/// Typestate marker: a required [`StridedTileSource`] field has been set.
pub struct Set;
/// Typestate marker: a required [`StridedTileSource`] field is still missing.
pub struct Unset;

/// The fields an [`StridedTileSource`] accumulates; the typestate lives in the wrapper, not here.
struct TileSourceData<'a, R: Runtime> {
    binding: TensorBinding<R>,
    space: Option<&'a Space>,
    /// The concrete (real-extent) space, when minted by a [`Launcher`](crate::Launcher):
    /// lets [`build`](StridedTileSource::build) derive the bounds-check from overhang.
    concrete: Option<&'a Space>,
    subspace: &'a [Axis],
    batch_axes: &'a [Axis],
    /// How the subspace axes are storage-tiled in the binding; `None` is untiled.
    tiling: Option<StorageTiling>,
    /// The operand's own affine mapping, when it states one ([`gathered`](StridedTileSource::gathered));
    /// `None` derives it from the labeled dims instead.
    projection: Option<Projection>,
    v: usize,
    boundary: Option<Option<Boundary>>,
    stage: Option<StageStorage>,
    /// The launch's cube size (units per cube); set by [`Launcher::arg`](crate::Launcher::arg).
    units: usize,
    /// Present when the operand is quantized; [`realize`](StridedTileSource::realize) validates it.
    quant: Option<Quantization<R>>,
    /// What this operand is at the instruction; default [`Leaf::Memory`], the memory form.
    leaf: Leaf,
}

/// Typestate builder for a strided tile kernel operand, started with
/// [`Launcher::arg`](crate::Launcher::arg) or [`StridedOperand::source`]. The `Sp`/`Sub`
/// markers make [`build`](Self::build) exist only once both required setters are [`Set`];
/// `Sub` is satisfied by [`subspace`](Self::subspace), which labels the buffer's dims, or by
/// [`gathered`](Self::gathered), which states the affine mapping outright;
/// the `Q` marker records whether [`quantized`](Self::quantized) was called, so `build`
/// returns a [`StridedOperand`] (plain) or a [`QuantOperand`] (quantized) and no call
/// site ever probes an option.
pub struct StridedTileSource<'a, Sp, Sub, Q, R: Runtime> {
    data: TileSourceData<'a, R>,
    _state: PhantomData<(Sp, Sub, Q)>,
}

impl<'a, R: Runtime> StridedTileSource<'a, Unset, Unset, Unset, R> {
    pub(crate) fn new(binding: TensorBinding<R>) -> Self {
        StridedTileSource {
            data: TileSourceData {
                binding,
                space: None,
                concrete: None,
                subspace: &[],
                batch_axes: &[],
                tiling: None,
                projection: None,
                v: 1,
                boundary: None,
                stage: None,
                units: 0,
                quant: None,
                leaf: Leaf::Memory,
            },
            _state: PhantomData,
        }
    }
}

impl<'a, Sp, Sub, Q, R: Runtime> StridedTileSource<'a, Sp, Sub, Q, R> {
    /// The global iteration space this operand projects from (required).
    pub fn space(mut self, space: &'a Space) -> StridedTileSource<'a, Set, Sub, Q, R> {
        self.data.space = Some(space);
        StridedTileSource {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// The inner block of axes the operand iterates, its `[row, col]` for a matmul (required
    /// unless [`gathered`](Self::gathered) states the mapping instead, non-empty). Complementary
    /// to [`batches`](Self::batches), the outer dims.
    pub fn subspace(mut self, axes: &'a [Axis]) -> StridedTileSource<'a, Sp, Set, Q, R> {
        self.data.subspace = axes;
        StridedTileSource {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// Sets an explicit affine [`Projection`] for a gathered operand (e.g. convolution or resample),
    /// mapping logical axes to buffer dimensions.
    ///
    /// Mutually exclusive with [`subspace`](Self::subspace), [`batches`](Self::batches), and
    /// [`tiling`](Self::tiling).
    ///
    /// # Bounds Checking & Extents
    /// - Boundary checking is enabled by default if the projection [`may_underflow`](Projection::may_underflow)
    ///   (override with [`checked`](Self::checked)). A window running off the buffer's *tail* is
    ///   not detected: nothing here knows how far the receptive field reaches, only the tiling's
    ///   own divisibility. A gather whose last window overruns the buffer, which a rational
    ///   mapping's does by construction, must state [`checked(true)`](Self::checked) itself.
    /// - An axis sharing a physical dim with another has no extent of its own here (the buffer holds
    ///   the receptive field they reach over), so if it is [`Dynamic`](crate::Extent) some other
    ///   operand of the operation must state its size ([`Tile::witnesses`](crate::Tile::witnesses)).
    ///   Nothing here can see the other operands, so an axis no operand answers for is reported at
    ///   expansion, by the op that walks it.
    /// - Dynamic scales, divisors and offsets ([`Scale::Dynamic`](crate::Scale),
    ///   [`Divisor::Dynamic`](crate::Divisor), [`Offset::Dynamic`](crate::Offset)) are passed at
    ///   runtime via [`TileArg::tile_gathered`](crate::TileArg::tile_gathered). A scale or divisor
    ///   declares a bound beside its runtime value, so the window it spans still has a comptime
    ///   size and the operand stages like any other; the launch must then stay within that bound.
    pub fn gathered(mut self, projection: Projection) -> StridedTileSource<'a, Sp, Set, Q, R> {
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

    /// Force the overhang bounds-check on or off (using [`Boundary::Zero`] when checked).
    /// Default: derived from the concrete space when minted by a [`Launcher`](crate::Launcher)
    /// (checked when an axis this operand addresses [`overhangs`](Space::overhangs), or when a
    /// [`gathered`](Self::gathered) mapping may underflow), else `Some(Boundary::Zero)`.
    /// A boolean convenience over [`with_boundary`](Self::with_boundary): it unconditionally
    /// overwrites whatever mode was set before it, so a `with_boundary(Some(Boundary::Clamp))`
    /// before this call is silently dropped back to `Zero`. Sequence a `Clamp` override after
    /// `checked`, not before it.
    pub fn checked(mut self, check: bool) -> Self {
        self.data.boundary = Some(if check { Some(Boundary::Zero) } else { None });
        self
    }

    /// Set the boundary handling mode for out-of-bounds access explicitly (`None` forces
    /// unchecked, `Some` forces checked at that mode).
    pub fn with_boundary(mut self, boundary: Option<Boundary>) -> Self {
        self.data.boundary = Some(boundary);
        self
    }

    /// What this operand is at the instruction: a memory window ([`Leaf::Memory`], the default)
    /// or a plane fragment in one of the two encodings. The partitioning says nothing about it;
    /// operands that disagree meet the kind-pairing panics at the instruction.
    pub fn leaf(mut self, leaf: Leaf) -> Self {
        self.data.leaf = leaf;
        self
    }

    /// The [`StageStorage`] layout of the smem stages derived from this operand. Default
    /// [`StageStorage::for_leaf`]: storage-tiled for a cmma leaf, plain strided otherwise.
    pub fn stage(mut self, stage: StageStorage) -> Self {
        self.data.stage = Some(stage);
        self
    }

    /// The concrete (real-extent) space the bounds-check derives from; set by
    /// [`Launcher::arg`](crate::Launcher::arg).
    pub(crate) fn concrete(mut self, space: &'a Space) -> Self {
        self.data.concrete = Some(space);
        self
    }

    /// The launch's cube size (units per cube); set by [`Launcher::arg`](crate::Launcher::arg).
    pub(crate) fn cube_units(mut self, units: usize) -> Self {
        self.data.units = units;
        self
    }
}

impl<'a, Sp, Sub, R: Runtime> StridedTileSource<'a, Sp, Sub, Unset, R> {
    /// Mark the operand as quantized: its binding holds the scheme's storage element (declared
    /// **in values**; a packed store's buffer is narrower than its shape by the packing
    /// factor), and `scales` + `scheme` let reads dequantize into the kernel's served type.
    /// [`vectorize`](Self::vectorize) still names the *served* width. `dequant_at` says how far the
    /// quantized form travels; it rides here rather than in its own setter because the quantized
    /// form ends at exactly one boundary, so one call says it once by construction. Which values
    /// are available is a capability of this operand's transports, and [`build`](Self::build)
    /// refuses one nothing can honour. Flips the typestate: `build` now yields a [`QuantOperand`].
    pub fn quantized(
        mut self,
        scales: TensorArg<R>,
        scheme: QuantScheme,
        dequant_at: DequantAt,
    ) -> StridedTileSource<'a, Sp, Sub, Set, R> {
        self.data.quant = Some(Quantization::new(scales, scheme, dequant_at));
        StridedTileSource {
            data: self.data,
            _state: PhantomData,
        }
    }
}

/// How an operand is quantized: the scales beside its values, the scheme saying how to fold them
/// back in, and how far the quantized form travels before something decodes it. One thing, because
/// none of the three says anything on its own — a scheme without scales cannot be applied, and an
/// [`DequantAt`] without a scheme has nothing to bound.
pub struct Quantization<R: Runtime> {
    pub scales: TensorArg<R>,
    pub scheme: QuantScheme,
    pub dequant_at: DequantAt,
}

impl<R: Runtime> Quantization<R> {
    pub fn new(scales: TensorArg<R>, scheme: QuantScheme, dequant_at: DequantAt) -> Self {
        Quantization {
            scales,
            scheme,
            dequant_at,
        }
    }

    /// Values per stored element: `1` unless the scheme packs several into each.
    pub fn num_quants(&self) -> usize {
        self.scheme.num_quants()
    }

    /// Refuse what this quantization cannot serve, on the caller's thread: the scheme against the
    /// operand's cuts and served width, the [`DequantAt`] against the reader that would have to honour
    /// it. Both rules live here because both are facts about this quantization and nothing else.
    pub(crate) fn validate(&self, space: &Space, vector_size: usize, leaf: Leaf) {
        validate_scheme(space, vector_size, self.scheme);
        validate_dequant_at(self.dequant_at, leaf);
    }
}

/// A built plain operand: the tensor argument, its comptime [`TileSpec`], and the served
/// width (also the binding width: the launch value for its `Size` generic).
pub struct StridedOperand<R: Runtime> {
    pub tensor: TensorArg<R>,
    pub vector_size: usize,
    pub spec: TileSpec,
}

impl<R: Runtime> StridedOperand<R> {
    /// The operand as the kernel's [`TileArg`](crate::TileArg) launch argument.
    pub fn arg<E: Numeric, V: Size>(self) -> TileArgLaunch<'static, E, V, R> {
        TileArgLaunch::new(self.tensor, self.spec)
    }
}

/// A built quantized operand: the storage-typed tensor and its spec, plus the scales and
/// the comptime scheme as first-class fields; nothing to probe or drain at the call site.
pub struct QuantOperand<R: Runtime> {
    pub tensor: TensorArg<R>,
    /// Served width (values per line); the binding is narrower by the packing factor.
    pub vector_size: usize,
    pub spec: TileSpec,
    pub quant: Quantization<R>,
}

impl<R: Runtime> QuantOperand<R> {
    /// The width the binding is typed at: the launch value for the kernel's `Size`
    /// generic. A packed store's buffer is narrower than the served width by the packing
    /// factor ([`tile_dequant`](crate::TileArg::tile_dequant) serves binding width × pack).
    pub fn bound_width(&self) -> usize {
        self.vector_size / self.quant.num_quants()
    }

    /// The operand as the kernel's [`QuantTileArg`](crate::QuantTileArg) launch argument:
    /// values, scales, spec and scheme as one thing. Read
    /// [`bound_width`](Self::bound_width) before consuming.
    pub fn arg<E: Numeric, V: Size>(self) -> QuantTileArgLaunch<'static, E, V, R> {
        QuantTileArgLaunch::new(
            self.tensor,
            self.quant.scales,
            self.spec,
            self.quant.scheme,
            self.quant.dequant_at,
        )
    }
}

impl<R: Runtime> StridedOperand<R> {
    /// Start describing a strided tile kernel operand sourced from `binding`: a
    /// [`StridedTileSource`] builder. Set the required [`space`](StridedTileSource::space)
    /// and either [`subspace`](StridedTileSource::subspace) or
    /// [`gathered`](StridedTileSource::gathered) (`build` won't compile until both are
    /// set), then optionally [`batches`](StridedTileSource::batches),
    /// [`tiling`](StridedTileSource::tiling), [`vectorize`](StridedTileSource::vectorize),
    /// or [`checked`](StridedTileSource::checked). Optional defaults are the safe ones, so
    /// a forgotten optional setter degrades performance, never correctness.
    pub fn source<'a>(binding: TensorBinding<R>) -> StridedTileSource<'a, Unset, Unset, Unset, R> {
        StridedTileSource::new(binding)
    }
}

/// [`realize`](StridedTileSource::realize)'s product, consumed by the two typed builds.
struct Realized<R: Runtime> {
    tensor: TensorArg<R>,
    vector_size: usize,
    spec: TileSpec,
    quant: Option<Quantization<R>>,
}

impl<'a, Q, R: Runtime> StridedTileSource<'a, Set, Set, Q, R> {
    /// The derivation both builds share: settle the operand's [`Projection`] (the labeled dims
    /// folded into a [`ConcreteLayout`], or the [`gathered`](StridedTileSource::gathered) mapping
    /// as given), derive the bounds-check, and mint the comptime [`TileSpec`].
    fn realize(self) -> Realized<R> {
        let TileSourceData {
            mut binding,
            space,
            concrete,
            batch_axes,
            subspace,
            tiling,
            projection,
            v,
            boundary,
            stage,
            units,
            quant,
            leaf,
        } = self.data;
        let space = space.unwrap();

        // Use the explicit projection if gathered, or derive it from labeled axes.
        // `addressed` contains all logical axes used for bounds checking.
        let (projection, addressed) = match projection {
            Some(projection) => {
                check_stated(&binding, space, &projection, subspace, batch_axes, &tiling);
                let addressed = projection.logical_axes().to_vec();
                (projection, addressed)
            }
            None => labeled(&mut binding, subspace, batch_axes, tiling),
        };

        // Derive boundary check: use explicit override if set, otherwise check for overhang or underflow.
        let boundary = boundary.unwrap_or_else(|| match concrete {
            Some(concrete) => {
                let overhangs = addressed
                    .iter()
                    .filter(|&&axis| concrete.contains(axis))
                    .any(|&axis| concrete.overhangs(axis));
                (overhangs || projection.may_underflow()).then_some(Boundary::Zero)
            }
            None => Some(Boundary::Zero),
        });

        // Bounds-checked operands cannot be vectorized.
        assert!(
            !(boundary.is_some() && v > 1),
            "StridedTileSource: a bounds-checked operand cannot be vectorized; serve it scalar or \
             state `checked(false)` if the launch proves every access in bounds"
        );
        // Validate projection vector width alignment on the host side.
        projection.validate(v);

        let mut spec = TileSpec::new(projection)
            .with_boundary(boundary)
            .units(units)
            .leaf(leaf);
        if let Some(stage) = stage {
            spec = spec.staged(stage);
        }
        if let Some(quant) = &quant {
            // Quantization is not supported for gathered operands.
            assert!(
                spec.projection.untiled().is_direct(),
                "StridedTileSource::quantized: a gathered operand cannot be quantized; its scale \
                 grid is shaped over its logical axes, which its buffer's dims no longer match"
            );
            quant.validate(&space.project(spec.axes()), v, leaf);
        }
        Realized {
            tensor: binding.into_tensor_arg(),
            vector_size: v,
            spec,
            quant,
        }
    }
}

/// Derives a [`Projection`] from labeled subspace and batch axes.
///
/// Leading batch dimensions align with `batch_axes` (size-1 broadcast dims are omitted).
/// The inner subspace axes are laid out according to `tiling`.
///
/// Returns the projection along with all physical axes prior to broadcast omission (for bounds checking).
fn labeled<R: Runtime>(
    binding: &mut TensorBinding<R>,
    subspace: &[Axis],
    batch_axes: &[Axis],
    tiling: Option<StorageTiling>,
) -> (Projection, Vec<Axis>) {
    let rank = binding.shape.len();
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
        "StridedTileSource: binding rank {rank} is smaller than its subspace block of {block_dims} dims ({} axes over {tiling:?})",
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
    let mut shape = Vec::new();
    let mut strides = Vec::new();

    for (i, &axis) in physical_axes.iter().enumerate() {
        let extent = binding.shape[i];
        // A subspace axis never drops out, however small, since the tile is shaped over it.
        if batch_axes.contains(&axis) && extent == 1 && !subspace.contains(&axis) {
            continue;
        }
        phys.push(PhysicalAxis::new(axis, extent));
        shape.push(extent);
        strides.push(binding.strides[i]);
    }

    binding.shape = shape[..].into();
    binding.strides = strides[..].into();
    (
        Projection::of_layout(&ConcreteLayout::new(&phys)),
        physical_axes,
    )
}

/// Validates an explicit gathered [`Projection`] against the tensor binding and iteration space.
/// Ensures mutually exclusive labeling options (`subspace`, `batch_axes`, `tiling`) were not provided.
fn check_stated<R: Runtime>(
    binding: &TensorBinding<R>,
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
        binding.shape.len(),
        "StridedTileSource::gathered: the mapping addresses {} dims but the binding has {}",
        projection.physical_rank(),
        binding.shape.len()
    );
    for &axis in projection.logical_axes() {
        assert!(
            space.contains(axis),
            "StridedTileSource::gathered: the mapping spans {axis:?}, which the launched space \
             does not have"
        );
    }
}

impl<'a, R: Runtime> StridedTileSource<'a, Set, Set, Unset, R> {
    /// Build the plain operand; the operand ships as a plain `TensorArg` plus its
    /// comptime [`TileSpec`].
    pub fn build(self) -> StridedOperand<R> {
        let Realized {
            tensor,
            vector_size,
            spec,
            ..
        } = self.realize();
        StridedOperand {
            tensor,
            vector_size,
            spec,
        }
    }
}

impl<'a, Q, R: Runtime> StridedTileSource<'a, Set, Set, Q, R> {
    /// Build the quantized operand: the plain derivation plus its validated [`Quantization`].
    fn build_quant(self) -> QuantOperand<R> {
        let Realized {
            tensor,
            vector_size,
            spec,
            quant,
        } = self.realize();
        QuantOperand {
            tensor,
            vector_size,
            spec,
            quant: quant.unwrap(),
        }
    }
}

impl<'a, R: Runtime> StridedTileSource<'a, Set, Set, Set, R> {
    /// Build the quantized operand.
    pub fn build(self) -> QuantOperand<R> {
        self.build_quant()
    }
}

/// Refuse a [`DequantAt`] nothing can honour. Called by [`build`](StridedTileSource::build) so a bad
/// plan fails on the caller's thread, and again by [`Tile::of_dequant`](crate::Tile::of_dequant),
/// which every launch path reaches including the raw
/// [`QuantTileArgLaunch`](crate::QuantTileArgLaunch) one.
///
/// Both transports constrain, but only the leaf can differ here: this operand is
/// [`Delivery::Strided`](crate::Delivery) by construction (that is what a [`StridedTileSource`] is),
/// and a strided load runs code per element, so it decodes whatever it moves. Only the leaf is left:
/// a fragment load takes a raw window at one element type, so a leaf that loads fragments needs its
/// values already served. A [`Delivery::Tma`](crate::Delivery) operand would invert this — a bulk
/// copy moves raw bytes, so its stage keeps the stored form and [`DequantAt::Read`] is the only site
/// it can honour — but a tensor map carries no scales ([`TmaTileArg`](crate::TmaTileArg)), so a
/// quantized TMA operand is not expressible and the rule has no site to fire at yet.
pub(crate) fn validate_dequant_at(dequant_at: DequantAt, leaf: Leaf) {
    match (dequant_at, leaf) {
        (DequantAt::Load, _) => {}
        // The memory leaf reads through a matrix view; so does the manual-mma fragment load, which
        // addresses one element at a time. Only the intrinsic transports are opaque.
        (DequantAt::Read, Leaf::Memory) => {}
        (DequantAt::Read, Leaf::Mma { io, .. }) => assert!(
            matches!(io.lhs_load_method, LoadMethod::Manual)
                && matches!(io.rhs_load_method, LoadMethod::Manual),
            "DequantAt::Read: the ldmatrix transport copies raw lanes, so it cannot decode as it \
             reads; such an operand must be served by its load (DequantAt::Load)"
        ),
        (DequantAt::Read, other) => panic!(
            "DequantAt::Read: {other:?} loads fragments at one element type, so it cannot decode as \
             it reads; such an operand must be served by its load (DequantAt::Load)"
        ),
    }
}
