//! The tile-loading API: the one place a launched tensor becomes a [`TileArgLaunch`]. Every client
//! (matmul, dequantize, …) loads tiles through these constructors, so the layout/broadcast wiring
//! lives here, not at each call site.

use core::marker::PhantomData;

use cubecl::prelude::*;

use crate::{Axis, ConcreteLayout, PhysicalAxis, Space, Storage, TileArgLaunch};

/// A realized physical layout maps straight to a tile [`Storage`]: its passthrough (batch) prefix
/// is `start_axis`, its storage-tiling depth is `levels`.
impl From<&ConcreteLayout> for Storage {
    fn from(layout: &ConcreteLayout) -> Self {
        Storage::passthrough(layout.passthrough(), layout.levels())
    }
}

impl<E: Numeric, R: Runtime> TileArgLaunch<'static, E, R> {
    /// Start describing a strided tile kernel argument sourced from `binding` — a [`TileSource`]
    /// builder. Set the two required parts — the [`space`](TileSource::space) it projects from and the
    /// [`subspace`](TileSource::subspace) block it iterates (`build` won't compile until both are set) —
    /// then optionally complementary outer [`batches`](TileSource::batches), a
    /// [`vectorize`](TileSource::vectorize) line size, or opt out of the bounds-check
    /// ([`checked`](TileSource::checked)). Optional defaults are the safe ones — scalar, batchless,
    /// checked — so a forgotten *optional* setter degrades performance, never correctness.
    pub fn source<'a>(binding: TensorBinding<R>) -> TileSource<'a, Unset, Unset, E, R> {
        TileSource {
            data: TileSourceData {
                binding,
                space: None,
                subspace: &[],
                batch_axes: &[],
                v: 1,
                check: true,
                _ty: PhantomData,
            },
            _state: PhantomData,
        }
    }

    /// Load a strided operand from its realized [`ConcreteLayout`]: derive the spanned axes
    /// ([`distinct_axes`](ConcreteLayout::distinct_axes)) and the tiling [`Storage`] from the layout,
    /// and project `space` onto those axes. The innermost (`cols`) axis is served as `Vector<E, v>`
    /// lines — the re-lining happens in-kernel from the comptime `vector_size`, so the scalar
    /// buffer's shape/strides pass through untouched. The matmul-agnostic loader — the `layout`'s axes are in
    /// the binding's dim order — so a client just builds the operand's layout and hands it here.
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

    /// Load a strided global tensor as a tile served in `vector_size`-wide lines. Its
    /// `[pre…, grid…, tile…]` buffer is tiled in-kernel over `space` (the [`Tile`](crate::Tile) reads
    /// the physical shape/strides off the tensor). The [`Storage`] carries the tiling depth and the
    /// overhang bounds-check.
    pub fn strided(
        tensor: TensorArg<R>,
        vector_size: usize,
        space: Space,
        storage: Storage,
    ) -> Self {
        Self::new(tensor, vector_size, space, storage)
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
    subspace: &'a [Axis],
    batch_axes: &'a [Axis],
    v: usize,
    check: bool,
    _ty: PhantomData<E>,
}

/// Typestate builder for a strided tile kernel argument, started with [`TileArgLaunch::source`]. The
/// argument occupies a subspace of the global space, named by two complementary axis groups: the
/// inner [`subspace`](Self::subspace) block (the tile it iterates — its trailing buffer dims,
/// storage-tiled so labels repeat level-major: dim `i` is `subspace[i % subspace.len()]`) and the
/// outer [`batches`](Self::batches) (its leading dims, one axis each, dropped when size 1 — numpy
/// broadcast omission). The binding is set at construction; the `Sp`/`Sub` markers track the two
/// remaining required setters, so [`build`](Self::build) exists only once both [`space`](Self::space)
/// and [`subspace`](Self::subspace) are [`Set`]. Borrows the axis slices + `space` for the chain.
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

    /// The outer (batch) axes, complementary to the [`subspace`](Self::subspace) block: one per
    /// leading buffer dim, dropped when size 1 (numpy broadcast). Default none (unbatched).
    pub fn batches(mut self, axes: &'a [Axis]) -> Self {
        self.data.batch_axes = axes;
        self
    }

    /// Line the innermost axis as `Vector<E, v>` (default `1`, i.e. scalar). Only valid when that
    /// axis is contiguous.
    pub fn vectorize(mut self, v: usize) -> Self {
        self.data.v = v;
        self
    }

    /// Bounds-check the operand's overhang against `space` (default `true`); pass `false` to skip the
    /// check when the tiling is known to divide evenly.
    pub fn checked(mut self, check: bool) -> Self {
        self.data.check = check;
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
            batch_axes,
            subspace,
            v,
            check,
            ..
        } = self.data;
        let space = space.unwrap();
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

        let n = subspace.len();
        let block = binding.shape[batch_axes.len()..]
            .iter()
            .zip(&binding.strides[batch_axes.len()..])
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
