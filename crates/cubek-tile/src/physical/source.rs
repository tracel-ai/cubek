//! The [`StridedTileSource`] builder: the one place a launched tensor becomes a
//! [`StridedOperand`]. Every client (matmul, dequantize, …) loads tiles through it, so
//! the layout/broadcast wiring lives here, not at each call site.

use core::marker::PhantomData;

use cubecl::prelude::*;

use cubecl::quant::scheme::QuantScheme;

use crate::{
    Axis, ConcreteLayout, PhysicalAxis, QuantTileArgLaunch, Space, StageStorage, Storage,
    TileArgLaunch, TileSpec, validate_scheme,
};

/// A realized physical layout maps straight to a tile [`Storage`]: its passthrough (batch) prefix
/// is `start_axis`, its storage-tiling depth is `levels`.
impl From<&ConcreteLayout> for Storage {
    fn from(layout: &ConcreteLayout) -> Self {
        Storage::passthrough(layout.passthrough(), layout.levels())
    }
}

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
    levels: usize,
    v: usize,
    check: Option<bool>,
    stage: Option<StageStorage>,
    /// The launch's cube size (units per cube); set by [`Launcher::arg`](crate::Launcher::arg).
    units: usize,
    /// Quantization side-channel: the scales plus the scheme saying how to fold them back in.
    /// [`build`](StridedTileSource::build) validates the scheme and carries the pair through.
    quant: Option<(TensorArg<R>, QuantScheme)>,
}

/// Typestate builder for a strided tile kernel operand, started with
/// [`Launcher::arg`](crate::Launcher::arg) or [`StridedOperand::source`]. The `Sp`/`Sub`
/// markers make [`build`](Self::build) exist only once both required setters are [`Set`];
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
                levels: 0,
                v: 1,
                check: None,
                stage: None,
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

    /// The inner block of axes the operand iterates, its `[row, col]` for a matmul (required,
    /// non-empty). Complementary to [`batches`](Self::batches), the outer dims.
    pub fn subspace(mut self, axes: &'a [Axis]) -> StridedTileSource<'a, Sp, Set, Q, R> {
        self.data.subspace = axes;
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

    /// Storage-tiling depth: `levels` nested `[grid…, leaf]` splits per subspace axis, so the
    /// trailing block is `subspace × (levels + 1)` buffer dims. Default `0` (plain strided).
    pub fn levels(mut self, levels: usize) -> Self {
        self.data.levels = levels;
        self
    }

    /// Serve the innermost axis in `v`-wide lines (default `1`, i.e. scalar). Only valid when
    /// that axis is contiguous. The kernel's element type carries the width (`Vector<E, V>`).
    pub fn vectorize(mut self, v: usize) -> Self {
        self.data.v = v;
        self
    }

    /// Force the overhang bounds-check on or off. Default: derived from the concrete space when
    /// minted by a [`Launcher`](crate::Launcher) (checked exactly when a subspace axis
    /// [`overhangs`](Space::overhangs)), else `true`.
    pub fn checked(mut self, check: bool) -> Self {
        self.data.check = Some(check);
        self
    }

    /// The [`StageStorage`] layout of the smem stages derived from this operand. Default
    /// [`StageStorage::for_space`]: storage-tiled for a cmma leaf, plain strided otherwise.
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
    /// [`vectorize`](Self::vectorize) still names the *served* width. Flips the typestate:
    /// [`build`](Self::build) now yields a [`QuantOperand`].
    pub fn quantized(
        mut self,
        scales: TensorArg<R>,
        scheme: QuantScheme,
    ) -> StridedTileSource<'a, Sp, Sub, Set, R> {
        self.data.quant = Some((scales, scheme));
        StridedTileSource {
            data: self.data,
            _state: PhantomData,
        }
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
    pub scales: TensorArg<R>,
    pub scheme: QuantScheme,
}

impl<R: Runtime> QuantOperand<R> {
    /// The width the binding is typed at: the launch value for the kernel's `Size`
    /// generic. A packed store's buffer is narrower than the served width by the packing
    /// factor ([`tile_dequant`](crate::TileArg::tile_dequant) serves binding width × pack).
    pub fn bound_width(&self) -> usize {
        self.vector_size / self.scheme.num_quants()
    }

    /// The operand as the kernel's [`QuantTileArg`](crate::QuantTileArg) launch argument:
    /// values, scales, spec and scheme as one thing. Read
    /// [`bound_width`](Self::bound_width) before consuming.
    pub fn arg<E: Numeric, V: Size>(self) -> QuantTileArgLaunch<'static, E, V, R> {
        QuantTileArgLaunch::new(self.tensor, self.scales, self.spec, self.scheme)
    }
}

impl<R: Runtime> StridedOperand<R> {
    /// Start describing a strided tile kernel operand sourced from `binding`: a
    /// [`StridedTileSource`] builder. Set the required [`space`](StridedTileSource::space)
    /// and [`subspace`](StridedTileSource::subspace) (`build` won't compile until both are
    /// set), then optionally [`batches`](StridedTileSource::batches),
    /// [`levels`](StridedTileSource::levels), [`vectorize`](StridedTileSource::vectorize),
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
    quant: Option<(TensorArg<R>, QuantScheme)>,
}

impl<'a, Q, R: Runtime> StridedTileSource<'a, Set, Set, Q, R> {
    /// The derivation both builds share: fold the labeled dims into a [`ConcreteLayout`],
    /// derive the bounds-check from overhang, and mint the comptime [`TileSpec`] via
    /// [`TileSpec::from_concrete`].
    fn realize(self) -> Realized<R> {
        let TileSourceData {
            mut binding,
            space,
            concrete,
            batch_axes,
            subspace,
            levels,
            v,
            check,
            stage,
            units,
            quant,
        } = self.data;
        let space = space.unwrap();

        // The trailing block is `subspace × (levels + 1)` buffer dims; whatever leads it is this
        // operand's batches, labeled by the trailing (right-aligned) slice of `batch_axes`.
        let n = subspace.len();
        let rank = binding.shape.len();
        let block_dims = n * (levels + 1);
        assert!(
            rank >= block_dims,
            "StridedTileSource: binding rank {rank} is smaller than its subspace block of {block_dims} dims ({n} axes, levels = {levels})"
        );
        let batch_dims = rank - block_dims;
        assert!(
            batch_dims <= batch_axes.len(),
            "StridedTileSource: {batch_dims} batch dims but only {} batch axes given",
            batch_axes.len()
        );
        let batch_axes = &batch_axes[batch_axes.len() - batch_dims..];

        // Explicit override wins; a Launcher-minted source derives the check from overhang, and
        // the free-standing path stays conservatively checked.
        let check = check.unwrap_or_else(|| match concrete {
            Some(concrete) => (subspace.iter().chain(batch_axes))
                // A batch axis absent from the space is a broadcast omission (its size-1
                // dim drops out below): nothing to overhang.
                .filter(|&&axis| concrete.contains(axis))
                .any(|&axis| concrete.overhangs(axis)),
            None => true,
        });
        // A masked access counts its length in lines and would clip valid rows, so a
        // bounds-checked operand must stay scalar.
        assert!(
            !(check && v > 1),
            "StridedTileSource: a bounds-checked operand cannot be vectorized"
        );

        let mut phys = Vec::new();
        let mut shape = Vec::new();
        let mut strides = Vec::new();

        for (i, &axis) in batch_axes.iter().enumerate() {
            let extent = binding.shape[i];
            if extent == 1 {
                continue; // broadcast omission: the dim and its axis both drop out
            }
            phys.push(PhysicalAxis::new(axis, extent));
            shape.push(extent);
            strides.push(binding.strides[i]);
        }

        let block = binding.shape[batch_dims..]
            .iter()
            .zip(&binding.strides[batch_dims..])
            .enumerate();
        for (i, (&extent, &stride)) in block {
            phys.push(PhysicalAxis::new(subspace[i % n], extent));
            shape.push(extent);
            strides.push(stride);
        }

        binding.shape = shape[..].into();
        binding.strides = strides[..].into();
        let mut spec = TileSpec::from_concrete(&ConcreteLayout::new(&phys), check, units);
        if let Some(stage) = stage {
            spec = spec.staged(stage);
        }
        if let Some((_, scheme)) = &quant {
            validate_scheme(&space.project(&spec.axes), v, *scheme);
        }
        Realized {
            tensor: binding.into_tensor_arg(),
            vector_size: v,
            spec,
            quant,
        }
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

impl<'a, R: Runtime> StridedTileSource<'a, Set, Set, Set, R> {
    /// Build the quantized operand: the same derivation plus the validated scheme, with
    /// the scales and scheme as first-class fields.
    pub fn build(self) -> QuantOperand<R> {
        let Realized {
            tensor,
            vector_size,
            spec,
            quant,
        } = self.realize();
        let (scales, scheme) = quant.unwrap();
        QuantOperand {
            tensor,
            vector_size,
            spec,
            scales,
            scheme,
        }
    }
}
