//! The [`StridedTileSource`] builder: the one place a launched tensor becomes a
//! [`StridedTileArgLaunch`]. Every client (matmul, dequantize, …) loads tiles through it, so
//! the layout/broadcast wiring lives here, not at each call site.

use core::marker::PhantomData;

use cubecl::prelude::*;

use cubecl::quant::scheme::QuantScheme;

use crate::{
    Axis, ConcreteLayout, PhysicalAxis, Space, StageStorage, Storage, StridedTileArgLaunch,
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
struct TileSourceData<'a, E, R: Runtime> {
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
    /// `build` binds the tensor at the scheme's storage width and attaches these.
    quant: Option<(TensorArg<R>, QuantScheme)>,
    _ty: PhantomData<E>,
}

/// Typestate builder for a strided tile kernel argument, started with
/// [`StridedTileArgLaunch::source`]. The `Sp`/`Sub` markers make [`build`](Self::build) exist
/// only once both required setters are [`Set`].
pub struct StridedTileSource<'a, Sp, Sub, E, R: Runtime> {
    data: TileSourceData<'a, E, R>,
    _state: PhantomData<(Sp, Sub)>,
}

impl<'a, E, R: Runtime> StridedTileSource<'a, Unset, Unset, E, R> {
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
                _ty: PhantomData,
            },
            _state: PhantomData,
        }
    }
}

impl<'a, Sp, Sub, E, R: Runtime> StridedTileSource<'a, Sp, Sub, E, R> {
    /// The global iteration space this argument projects from (required).
    pub fn space(mut self, space: &'a Space) -> StridedTileSource<'a, Set, Sub, E, R> {
        self.data.space = Some(space);
        StridedTileSource {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// The inner block of axes the operand iterates — its `[row, col]` for a matmul (required,
    /// non-empty). Complementary to [`batches`](Self::batches), the outer dims.
    pub fn subspace(mut self, axes: &'a [Axis]) -> StridedTileSource<'a, Sp, Set, E, R> {
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

    /// Line the innermost axis as `Vector<E, v>` (default `1`, i.e. scalar). Only valid when that
    /// axis is contiguous.
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

    /// Mark the operand as quantized: its binding holds the scheme's storage element (declared
    /// **in values** — a packed store's buffer is narrower than its shape by the packing
    /// factor), and `scales` + `scheme` let reads dequantize into the kernel's served type.
    /// [`vectorize`](Self::vectorize) still names the *served* width; `build` derives the
    /// storage width the tensor is truly bound at. Default not quantized.
    pub fn quantized(mut self, scales: TensorArg<R>, scheme: QuantScheme) -> Self {
        self.data.quant = Some((scales, scheme));
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

impl<'a, E: Numeric, R: Runtime> StridedTileSource<'a, Set, Set, E, R> {
    /// Build the operand's [`ConcreteLayout`] from its labeled dims and load it via
    /// [`from_concrete`](StridedTileArgLaunch::from_concrete). Available only once space and subspace are
    /// both set, so the `unwrap` below cannot fire.
    pub fn build(self) -> StridedTileArgLaunch<'static, E, R> {
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
            ..
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
        let mut arg = StridedTileArgLaunch::from_concrete(
            binding,
            &ConcreteLayout::new(&phys),
            space,
            v,
            check,
            units,
        );
        if let Some(stage) = stage {
            arg = arg.stage(stage);
        }
        if let Some((scales, scheme)) = quant {
            arg = arg.quantized(scales, scheme);
        }
        arg
    }
}
