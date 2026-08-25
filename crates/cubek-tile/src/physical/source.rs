//! The [`StridedTileSource`] builder: the one place a launched tensor becomes a
//! [`StridedOperand`]. Every client (matmul, dequantize, …) loads tiles through it, so
//! the layout/broadcast wiring lives here, not at each call site.

use core::marker::PhantomData;

use cubecl::prelude::*;

use cubecl::quant::scheme::QuantScheme;
use cubecl::std::tensor::layout::linear::linear_view;

use crate::{
    Axis, Boundary, ConcreteLayout, DequantAt, Instruction, LoadMethod, PhysicalAxis, Projection,
    QuantTileArgLaunch, Residence, Space, StageStorage, StorageTiling, TileArgLaunch, TileSpec,
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
    storage: Option<StageStorage>,
    /// Where the operand lives at each level of `space`, coarse to fine; empty stages nothing.
    residence: Vec<Residence>,
    /// The launch's cube size (units per cube); set by [`Launcher::arg`](crate::Launcher::arg).
    units: usize,
    /// Present when the operand is quantized; [`realize`](StridedTileSource::realize) validates it.
    quant: Option<Quantization<R>>,
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
                storage: None,
                residence: Vec::new(),
                units: 0,
                quant: None,
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
    ///
    /// `checked(false)` disarms the operand outright. `checked(true)` states the *mode*, not the
    /// axis list: [`build`](Self::build) still lands it only on the coordinate axes that can
    /// leave the buffer, so an axis whose map is the identity of a logical axis whose tiling
    /// divides stays unchecked whatever this says. That filtering is what lets a clamped operand
    /// keep a vectorized innermost axis; there is no setter that masks a settled axis.
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

    /// The [`StageStorage`] layout of the smem stages derived from this operand. Default
    /// [`StageStorage::for_stages`]: storage-tiled for a cmma operand, plain strided otherwise.
    pub fn storage(mut self, storage: StageStorage) -> Self {
        self.data.storage = Some(storage);
        self
    }

    /// Take the per-level residences from `operand`'s stages, stated where the levels were
    /// declared ([`Operand::stage`](crate::Operand::stage)) and checked against the space's
    /// depth by [`build`](Self::build). Default: every level [`InPlace`](Residence::InPlace),
    /// so an operand that stages nothing is read where it already is and the walk
    /// materializes nothing.
    ///
    /// Independent of the level's [`Buffering`](crate::Buffering): one operand may take a shared
    /// stage at a double-buffered level while another streams from global memory through it.
    pub fn operand(self, operand: &crate::Operand) -> Self {
        self.residence(&operand.residences())
    }

    /// [`operand`](Self::operand)'s raw form, for the bridges that already hold the column.
    pub(crate) fn residence(mut self, residence: &[Residence]) -> Self {
        self.data.residence = residence.to_vec();
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
    /// `scales` holds one binding per scheme level, innermost first.
    /// [`vectorize`](Self::vectorize) still names the *served* width. `dequant_at` says how far the
    /// quantized form travels; it rides here rather than in its own setter because the quantized
    /// form ends at exactly one boundary, so one call says it once by construction. Which values
    /// are available is a capability of this operand's transports, and [`build`](Self::build)
    /// refuses one nothing can honour. Flips the typestate: `build` now yields a [`QuantOperand`].
    pub fn quantized(
        mut self,
        scales: &[TensorBinding<R>],
        scheme: QuantScheme,
        dequant_at: DequantAt,
    ) -> StridedTileSource<'a, Sp, Sub, Set, R> {
        self.data.quant = Some(Quantization::new(scales, scheme, dequant_at));
        StridedTileSource {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// [`quantized`](Self::quantized) for a lookup scheme
    /// ([`QuantMode::Lookup`](cubecl::quant::scheme::QuantMode)): each stored field indexes
    /// `table` and a read reconstructs `table[field] * scale`. The table must hold `2^bits`
    /// f32 entries: the unpack's mask bounds every index to that range, so a shorter buffer
    /// is read out of bounds, and no check here can see its length.
    pub fn quantized_lookup(
        mut self,
        scales: TensorArg<R>,
        table: BufferArg<R>,
        scheme: QuantScheme,
        dequant_at: DequantAt,
    ) -> StridedTileSource<'a, Sp, Sub, Set, R> {
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
pub struct Quantization<R: Runtime> {
    /// The innermost level's scales, the only ones addressed per position.
    pub scales: TensorArg<R>,
    /// The global level's scale, one for the whole tensor, present exactly when the scheme has a
    /// second level ([`validate`](Self::validate) holds the two together).
    pub global: Option<TensorBinding<R>>,
    /// A lookup scheme's `2^bits`-entry table, present exactly under
    /// [`QuantMode::Lookup`](cubecl::quant::scheme::QuantMode);
    /// [`validate`](Self::validate) holds the two together.
    pub table: Option<BufferArg<R>>,
    pub scheme: QuantScheme,
    pub dequant_at: DequantAt,
}

impl<R: Runtime> Quantization<R> {
    /// `scales` holds one binding per scheme level, innermost first. Only the innermost level's
    /// scales are addressed per position; an global level covers the whole tensor, so its binding
    /// is read once from its first element. This only checks the slice holds 1 or 2 bindings;
    /// whether that count actually matches `scheme`'s own level count is [`validate`](Self::validate)'s
    /// job, via `check_scale_bindings`.
    pub fn new(scales: &[TensorBinding<R>], scheme: QuantScheme, dequant_at: DequantAt) -> Self {
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
        scales: TensorArg<R>,
        table: BufferArg<R>,
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
    /// operand's cuts and served width, the [`DequantAt`] against the reader that would have to honour
    /// it. Both rules live here because both are facts about this quantization and nothing else.
    pub(crate) fn validate(
        &self,
        space: &Space,
        vector_size: usize,
        register_stage: Option<Instruction>,
    ) {
        cubecl::std::quant::check_scale_bindings(&self.scheme, 1 + self.global.is_some() as usize);
        validate_scheme(space, vector_size, self.scheme);
        validate_dequant_at(self.dequant_at, register_stage);
        cubecl::std::quant::check_table_bindings(&self.scheme, self.table.is_some());
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
            self.quant.global.map(linear_view).into(),
            self.quant.table.into(),
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
    /// [`residence`](StridedTileSource::residence) or [`checked`](StridedTileSource::checked).
    /// Most optional defaults are conservative; residency defaults to reading in place, so an
    /// operand a fragment load cannot address must state where it is materialized.
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
            storage,
            residence,
            units,
            quant,
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

        projection.validate(v);
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
            Some(axis) => {
                concrete.is_some_and(|space| space.contains(axis) && !space.overhangs(axis))
            }
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

        // Validate that explicit residences match the space's depth: extra or missing levels lead
        // to silent misconfiguration.
        let depth = space.partitioner().depth();
        assert!(
            residence.is_empty() || residence.len() == depth,
            "StridedTileSource::residence: {} residences stated but the space has {depth} levels",
            residence.len()
        );

        let mut spec = TileSpec::new(projection)
            .boundaries(&boundaries)
            .units(units)
            .residence(&residence);
        if let Some(storage) = storage {
            spec = spec.storage(storage);
        }
        if let Some(quant) = &quant {
            // Quantization is not supported for gathered operands.
            assert!(
                spec.projection.untiled().is_direct(),
                "StridedTileSource::quantized: a gathered operand cannot be quantized; its scale \
                 grid is shaped over its logical axes, which its buffer's dims no longer match"
            );
            quant.validate(&space.project(spec.axes()), v, space.instruction());
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
/// Both transports constrain, but only the instruction form can differ here: this operand is
/// [`Delivery::Copy`](crate::Delivery) by construction (that is what a [`StridedTileSource`] is),
/// and a strided load runs code per element, so it decodes whatever it moves. Only the form is
/// left: a fragment load takes a raw window at one element type, so a form that loads fragments
/// needs its values already served. A [`Delivery::Tma`](crate::Delivery) operand would invert this:
/// a bulk copy moves raw bytes, so its stage keeps the stored form and [`DequantAt::Read`] is the
/// only site it can honour, but a tensor map carries no scales ([`TmaTileArg`](crate::TmaTileArg)),
/// so a quantized TMA operand is not expressible and the rule has no site to fire at yet.
pub(crate) fn validate_dequant_at(dequant_at: DequantAt, register_stage: Option<Instruction>) {
    match (dequant_at, register_stage) {
        (DequantAt::Load, _) => {}
        // A memory window is read through the quant-transparent matrix view; so is the
        // manual-mma fragment load, which addresses one element at a time. Only the intrinsic
        // transports are opaque.
        (DequantAt::Read, None | Some(Instruction::Registers { .. })) => {}
        (DequantAt::Read, Some(Instruction::Mma { io })) => assert!(
            matches!(io.lhs_load_method, LoadMethod::Manual)
                && matches!(io.rhs_load_method, LoadMethod::Manual),
            "DequantAt::Read: the ldmatrix transport copies raw lanes, so it cannot decode as it \
             reads; such an operand must be served by its load (DequantAt::Load)"
        ),
        (DequantAt::Read, Some(Instruction::Cmma)) => panic!(
            "DequantAt::Read: a cmma fragment loads at one element type, so it cannot decode as \
             it reads; such an operand must be served by its load (DequantAt::Load)"
        ),
    }
}
