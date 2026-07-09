//! The tile-loading API: the one place a launched tensor becomes a [`TileArgLaunch`]. Every client
//! (matmul, dequantize, …) loads tiles through these constructors, so the layout/broadcast wiring
//! lives here, not at each call site.

use core::marker::PhantomData;

use cubecl::prelude::*;

use cubecl::quant::scheme::QuantScheme;

use crate::{
    Axis, ConcreteLayout, PhysicalAxis, QuantArgLaunch, Space, Storage, TileArgLaunch, VecTensorArg,
};

/// A realized physical layout maps straight to a tile [`Storage`]: its passthrough (batch) prefix
/// is `start_axis`, its storage-tiling depth is `levels`.
impl From<&ConcreteLayout> for Storage {
    fn from(layout: &ConcreteLayout) -> Self {
        Storage::passthrough(layout.passthrough(), layout.levels())
    }
}

impl<E: Numeric, R: Runtime> TileArgLaunch<'static, E, R> {
    /// Start describing a strided tile kernel argument sourced from `binding`: a
    /// [`TileSource`] builder. Set the required [`space`](TileSource::space) and
    /// [`subspace`](TileSource::subspace) (`build` won't compile until both are set), then
    /// optionally [`batches`](TileSource::batches), [`levels`](TileSource::levels),
    /// [`vectorize`](TileSource::vectorize), or [`checked`](TileSource::checked).
    /// Optional defaults are the safe ones, so a forgotten optional setter degrades
    /// performance, never correctness.
    pub fn source<'a>(binding: TensorBinding<R>) -> TileSource<'a, Unset, Unset, E, R> {
        TileSource {
            data: TileSourceData {
                binding,
                space: None,
                concrete: None,
                subspace: &[],
                batch_axes: &[],
                levels: 0,
                v: 1,
                check: None,
                _ty: PhantomData,
            },
            _state: PhantomData,
        }
    }

    /// Load a strided operand from its realized [`ConcreteLayout`]: derive the spanned
    /// axes and the tiling [`Storage`] from the layout, and project `space` onto those
    /// axes. The innermost axis is served as `Vector<E, v>` lines, re-lined in-kernel so
    /// the scalar buffer's shape/strides pass through untouched.
    pub fn from_concrete(
        binding: TensorBinding<R>,
        layout: &ConcreteLayout,
        space: &Space,
        v: usize,
        check: bool,
    ) -> Self {
        Self::strided(
            binding.into_tensor_arg(),
            v,
            space.project(&layout.distinct_axes()),
            Storage::from(layout).checked(check),
        )
    }

    /// Load a strided global tensor as a tile served in `vector_size`-wide lines (the binding is
    /// typed `Vector<E, vector_size>`, see [`VecTensor`](crate::VecTensor)). Its
    /// `[pre…, grid…, tile…]` buffer is tiled in-kernel over `space` (the [`Tile`](crate::Tile)
    /// reads the physical shape/strides off the tensor). The [`Storage`] carries the tiling depth
    /// and the overhang bounds-check.
    pub fn strided(
        tensor: TensorArg<R>,
        vector_size: usize,
        space: Space,
        storage: Storage,
    ) -> Self {
        Self::new(
            VecTensorArg::new(tensor, vector_size),
            vector_size,
            space,
            storage,
            ComptimeOptionArgs::None,
        )
    }

    /// Mark the operand as quantized: its tensor holds the storage element, and `scales` +
    /// `scheme` let reads dequantize into the kernel's served type
    /// ([`tile_dequant`](crate::TileArg::tile_dequant)).
    pub fn quantized(mut self, scales: TensorArg<R>, scheme: QuantScheme) -> Self {
        self.quant = ComptimeOptionArgs::Some(QuantArgLaunch::new(scales, scheme));
        self
    }
}

/// Typestate marker: a required [`TileSource`] field has been set.
pub struct Set;
/// Typestate marker: a required [`TileSource`] field is still missing.
pub struct Unset;

/// The fields an [`TileSource`] accumulates; the typestate lives in the wrapper, not here.
struct TileSourceData<'a, E, R: Runtime> {
    binding: TensorBinding<R>,
    space: Option<&'a Space>,
    /// The concrete (real-extent) space, when minted by a [`Launcher`](crate::Launcher):
    /// lets [`build`](TileSource::build) derive the bounds-check from overhang.
    concrete: Option<&'a Space>,
    subspace: &'a [Axis],
    batch_axes: &'a [Axis],
    levels: usize,
    v: usize,
    check: Option<bool>,
    _ty: PhantomData<E>,
}

/// Typestate builder for a strided tile kernel argument, started with
/// [`TileArgLaunch::source`]. The `Sp`/`Sub` markers make [`build`](Self::build) exist
/// only once both required setters are [`Set`].
pub struct TileSource<'a, Sp, Sub, E, R: Runtime> {
    data: TileSourceData<'a, E, R>,
    _state: PhantomData<(Sp, Sub)>,
}

impl<'a, Sp, Sub, E, R: Runtime> TileSource<'a, Sp, Sub, E, R> {
    /// The global iteration space this argument projects from (required).
    pub fn space(mut self, space: &'a Space) -> TileSource<'a, Set, Sub, E, R> {
        self.data.space = Some(space);
        TileSource {
            data: self.data,
            _state: PhantomData,
        }
    }

    /// The inner block of axes the operand iterates — its `[row, col]` for a matmul (required,
    /// non-empty). Complementary to [`batches`](Self::batches), the outer dims.
    pub fn subspace(mut self, axes: &'a [Axis]) -> TileSource<'a, Sp, Set, E, R> {
        self.data.subspace = axes;
        TileSource {
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

    /// The concrete (real-extent) space the bounds-check derives from; set by
    /// [`Launcher::arg`](crate::Launcher::arg).
    pub(crate) fn concrete(mut self, space: &'a Space) -> Self {
        self.data.concrete = Some(space);
        self
    }
}

impl<'a, E: Numeric, R: Runtime> TileSource<'a, Set, Set, E, R> {
    /// Build the operand's [`ConcreteLayout`] from its labeled dims and load it via
    /// [`from_concrete`](TileArgLaunch::from_concrete). Available only once space and subspace are
    /// both set, so the `unwrap` below cannot fire.
    pub fn build(self) -> TileArgLaunch<'static, E, R> {
        let TileSourceData {
            mut binding,
            space,
            concrete,
            batch_axes,
            subspace,
            levels,
            v,
            check,
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
            "TileSource: binding rank {rank} is smaller than its subspace block of {block_dims} dims ({n} axes, levels = {levels})"
        );
        let batch_dims = rank - block_dims;
        assert!(
            batch_dims <= batch_axes.len(),
            "TileSource: {batch_dims} batch dims but only {} batch axes given",
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
            "TileSource: a bounds-checked operand cannot be vectorized"
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
        TileArgLaunch::from_concrete(binding, &ConcreteLayout::new(&phys), space, v, check)
    }
}
