//! The [`Tile`]: one operand's data — a [`Payload`] enum (`Gmem`/`Smem`/`Cmma`),
//! the comptime [`Space`] it projects, and a runtime window. The payload holds a
//! lifetime-free buffer; [`view`](Tile::view) rebuilds a [`ViewMut`] over it on
//! demand. A `Tile` is not launchable; a kernel takes a raw [`Tensor`] and wraps it
//! with [`from_tensor`](Tile::from_tensor).
//!
//! Vectorization is *transparent*: a `Tile` is size-free (`Tile<E>`), its buffer a
//! plain scalar slice (`Vector<E, Const<1>>`). An operand's line size is a
//! memory-layout property the launch boundary resolves — not a type parameter the
//! DSL threads — so `lhs`, `rhs`, and `out` make no assumption about each other's
//! vectorization. The contraction reads scalars (see [`contract`](Tile::contract)),
//! mirroring the production tile matmul.
use cubecl::{
    cmma::Matrix,
    prelude::*,
    std::tensor::{
        AsView, AsViewExpand, AsViewMut, AsViewMutExpand, View, ViewMut,
        layout::{Coords1d, CoordsDyn, Layout, LayoutExpand, tiled_view::TiledLayout},
    },
};

use super::*;

/// The physical storage of a launched tensor — how its `[pre…, grid…, tile…]`
/// buffer maps to the logical [`Space`]. `start_axis` leading axes are passthrough
/// (e.g. a batch the tiling skips); the rest are tiled with `levels` nesting. This
/// is a property of the *tensor* (its higher rank encodes the tiling), distinct from
/// the space's partitioner — and derived from the rank at the host, never written by
/// hand.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Storage {
    pub start_axis: usize,
    pub levels: usize,
}

impl Storage {
    /// Every axis tiled, no passthrough — `levels` read off the tensor's rank
    /// (`physical_rank / logical_rank - 1`).
    pub fn of(physical_rank: usize, logical_rank: usize) -> Self {
        Storage { start_axis: 0, levels: physical_rank / logical_rank - 1 }
    }

    /// `start_axis` leading axes passthrough, the rest tiled to `levels` depth.
    pub fn passthrough(start_axis: usize, levels: usize) -> Self {
        Storage { start_axis, levels }
    }
}

/// The launchable form of a [`Tile`]: a reference to the global [`Tensor`] plus the
/// comptime [`Space`] and [`Storage`] it's read with. One self-contained struct per
/// operand at the launch boundary — `let tile = arg.tile()` on the first line of the
/// kernel. (A `Tensor` is `?Sized` so it can't be held by value, but a `&Tensor`
/// can, which is why this is the launchable subset and [`Tile`] is not.)
#[derive(CubeType, CubeLaunch)]
pub struct TileArg<'a, E: Numeric> {
    pub tensor: &'a Tensor<Vector<E, Const<1>>>,
    #[cube(comptime)]
    pub space: Space,
    #[cube(comptime)]
    pub storage: Storage,
}

#[cube]
impl<'a, E: Numeric> TileArg<'a, E> {
    /// Wrap this launch arg into a `Gmem` [`Tile`] — the conversion the kernel does
    /// up front. See [`Tile::from_tensor`].
    pub fn tile(&self) -> Tile<E> {
        Tile::from_tensor(self.tensor, comptime!(self.space.clone()), comptime!(self.storage))
    }
}

/// One operand's data. The window (`origin`/`extent`) lives on the struct so
/// [`at`](Tile::at) edits it uniformly across payload kinds; the 2-D matrix a tile
/// collapses into is a bare `Coords2d` view ([`matrix`](Tile::matrix)).
#[derive(CubeType)]
pub struct Tile<E: Numeric> {
    pub payload: Payload<E>,
    /// Window origin in logical coordinates (accumulates across [`at`](Tile::at)s).
    pub origin: CoordsDyn,
    /// Window extent in logical coordinates (the current sub-tile's edges).
    pub extent: CoordsDyn,
    /// The space this tile projects — owns the partitioner.
    #[cube(comptime)]
    pub space: Space,
}

/// A tile's backing store. Every variant is lifetime-free (a `Box<[E]>` or a
/// [`cmma::Matrix`]); `Tensor`/`Shared` are `?Sized` and a `ViewMut` carries a
/// borrow, so neither can be stored — [`view`](Tile::view) rebuilds one on demand.
#[derive(CubeType)]
pub enum Payload<E: Numeric> {
    /// A global-tensor operand or the accumulator.
    Gmem(MemData<E>),
    /// A shared-memory staging buffer or smem accumulator.
    Smem(MemData<E>),
    /// A tensor-core fragment — register/MMA-resident, no memory address.
    Cmma(Matrix<E>),
}

/// The lifetime-erased buffer plus the physical shape/strides and tiling spec to
/// rebuild its [`GmemLayout`]. Fixed at construction, never recomputed from the
/// (dividing) `Space`, so a staged smem sub-tile keeps addressing the whole buffer
/// it was allocated for after [`at`](Tile::at) windows it down.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct MemData<E: Numeric> {
    buffer: Box<[Vector<E, Const<1>>]>,
    physical_shape: CoordsDyn,
    physical_strides: CoordsDyn,
    /// Leading `start_axis` axes are passthrough (e.g. a batch the tiling skips).
    #[cube(comptime)]
    start_axis: usize,
    /// Tiled axes, each split into `levels + 1` `[grid…, tile…]` parts.
    #[cube(comptime)]
    num_tiled: usize,
    /// Nested tile levels (`0` = smem / untiled).
    #[cube(comptime)]
    levels: usize,
}

#[cube]
impl<E: Numeric> Tile<E> {
    /// Wrap a launched [`Tensor`] into a whole `Gmem` tile: the operand's semantic
    /// [`Space`] and the tensor's physical [`Storage`] (two separate things). The
    /// kernel takes the tensor by `&` (it is `?Sized`) and we erase the borrow into a
    /// `Box`. The tensor is scalar (`Const<1>`): vectorization is resolved at the
    /// launch boundary, the DSL stays size-free.
    pub fn from_tensor(
        tensor: &Tensor<Vector<E, Const<1>>>,
        #[comptime] space: Space,
        #[comptime] storage: Storage,
    ) -> Tile<E> {
        let start_axis = comptime!(storage.start_axis);
        let num_tiled = comptime!(space.rank() - storage.start_axis);
        let levels = comptime!(storage.levels);
        let mut physical_shape = CoordsDyn::new();
        let mut physical_strides = CoordsDyn::new();
        #[unroll]
        for i in 0..comptime!(start_axis + (levels + 1) * num_tiled) {
            physical_shape.push(tensor.shape(i) as u32);
            physical_strides.push(tensor.stride(i) as u32);
        }
        let buffer = unsafe { tensor.as_slice().as_boxed_unchecked() };
        let (origin, extent) = full_window(comptime!(space.clone()));
        Tile::<E> {
            payload: Payload::new_Gmem(MemData::<E> {
                buffer,
                physical_shape,
                physical_strides,
                start_axis,
                num_tiled,
                levels,
            }),
            origin,
            extent,
            space: comptime!(space),
        }
    }

    /// Wrap a shared-memory buffer as a whole `Smem` tile — the [`stage`](Tile::stage)
    /// destination. Row-major over `space`; the borrow is erased into a `Box`.
    pub fn smem(smem: &Shared<[Vector<E, Const<1>>]>, #[comptime] space: Space) -> Tile<E> {
        let buffer = unsafe { smem.inner_ref().as_boxed_unchecked() };
        let (physical_shape, physical_strides) = row_major(comptime!(space.clone()));
        let (origin, extent) = full_window(comptime!(space.clone()));
        Tile::<E> {
            payload: Payload::new_Smem(MemData::<E> {
                buffer,
                physical_shape,
                physical_strides,
                start_axis: comptime!(0usize),
                num_tiled: comptime!(space.rank()),
                levels: comptime!(0usize),
            }),
            origin,
            extent,
            space: comptime!(space),
        }
    }

    /// A read [`View`] over this tile's data — the buffer re-viewed through its base
    /// layout, then the [`Window`]. Panics on a [`Cmma`](Payload::Cmma) tile.
    pub fn view(&self) -> View<'_, Vector<E, Const<1>>, CoordsDyn> {
        let window = Window::new(self.origin.clone(), self.extent.clone());
        match &self.payload {
            Payload::Gmem(g) => g.buffer.view(g.base()).view(window),
            Payload::Smem(g) => g.buffer.view(g.base()).view(window),
            Payload::Cmma(_) => panic!("Tile::view: a cmma fragment has no memory view"),
        }
    }

    /// A write [`ViewMut`] over this tile's data — the mutable twin of
    /// [`view`](Tile::view), for the accumulator (and a `stage` destination).
    pub fn view_mut(&mut self) -> ViewMut<'_, Vector<E, Const<1>>, CoordsDyn> {
        let window = Window::new(self.origin.clone(), self.extent.clone());
        match &mut self.payload {
            Payload::Gmem(g) => {
                let base = g.base();
                g.buffer.view_mut(base).view_mut(window)
            }
            Payload::Smem(g) => {
                let base = g.base();
                g.buffer.view_mut(base).view_mut(window)
            }
            Payload::Cmma(_) => panic!("Tile::view_mut: a cmma fragment has no memory view"),
        }
    }

    /// Window this tile down to `region` — the same axes at the sub-tile edges,
    /// shifted to the region's origin. No copy and no axis bookkeeping: the tile
    /// projects `region` onto its own axes, so `lhs ∈ {M,K}` and `out ∈ {M,N}` agree
    /// without the caller matching them. Keeps the payload kind at a smaller extent.
    pub fn at(&self, region: &Region) -> Tile<E> {
        let space = region.space();
        let mut origin = CoordsDyn::new();
        let mut extent = CoordsDyn::new();
        #[unroll]
        for p in 0..comptime!(self.space.rank()) {
            let axis = comptime!(self.space.axis_at(p));
            let edge = comptime!(self.space.partitioner().sub_tile_edge(axis));
            let index = project_flat(
                comptime!(self.space.levels(axis)),
                comptime!(space.levels(axis)),
                region.coord(axis),
            );
            // Accumulate onto the current window: child reads base[origin + index*edge + pos].
            origin.push(self.origin[p] + (index * edge) as u32);
            extent.push(comptime!(edge as i64) as u32);
        }
        // The sub-tile keeps the source's payload kind at a smaller extent (re-box
        // the same buffer — a `Box` handle copy onto the same memory).
        let payload = match &self.payload {
            Payload::Gmem(g) => Payload::new_Gmem(g.rebox()),
            Payload::Smem(g) => Payload::new_Smem(g.rebox()),
            Payload::Cmma(_) => panic!("Tile::at: a cmma fragment cannot be located by view"),
        };
        Tile::<E> {
            payload,
            origin,
            extent,
            space: comptime!(self.space.divide()),
        }
    }

    /// Stage `src` into `self` — an element copy across a level (e.g. gmem → smem).
    /// Unlike [`at`](Tile::at) (a within-level view) this moves data; sync after.
    pub fn stage(&mut self, src: &Tile<E>) {
        let matrices = self.matrix_count();
        for j in 0..matrices {
            let s = src.matrix(j);
            let mut d = self.matrix_mut(j);
            copy_2d::<E>(&mut d, &s);
        }
    }
}

#[cube]
impl<E: Numeric> MemData<E> {
    /// The flat-source base layout (logical [`CoordsDyn`] → buffer offset) from the
    /// stored *physical* shape/strides — the `[grid…, tile…]` split (gmem,
    /// `levels > 0`) or a plain strided dot (smem, `levels = 0`).
    fn base(&self) -> GmemLayout {
        GmemLayout {
            physical_shape: self.physical_shape.clone(),
            physical_strides: self.physical_strides.clone(),
            start_axis: comptime!(self.start_axis),
            num_tiled: comptime!(self.num_tiled),
            levels: comptime!(self.levels),
        }
    }

    /// Re-box onto the same buffer (a `Box` handle copy, same memory) with the same
    /// physical metadata — used by [`at`](Tile::at) to make a sub-tile.
    fn rebox(&self) -> MemData<E> {
        MemData::<E> {
            buffer: unsafe { self.buffer.as_boxed_unchecked() },
            physical_shape: self.physical_shape.clone(),
            physical_strides: self.physical_strides.clone(),
            start_axis: comptime!(self.start_axis),
            num_tiled: comptime!(self.num_tiled),
            levels: comptime!(self.levels),
        }
    }
}

/// The whole-tile window: `origin = 0`, `extent =` the space's per-axis extents.
#[cube]
fn full_window(#[comptime] space: Space) -> (CoordsDyn, CoordsDyn) {
    let mut origin = CoordsDyn::new();
    let mut extent = CoordsDyn::new();
    #[unroll]
    for p in 0..comptime!(space.rank()) {
        origin.push(u32::from_int(0));
        extent.push(u32::from_int(comptime!(space.extent(space.axis_at(p)) as i64)));
    }
    (origin, extent)
}

/// Row-major physical shape/strides over `space`'s per-axis extents — the smem
/// buffer's contiguous layout, stored in its [`MemData`] so it survives `at`'s
/// space division (see [`Tile::smem`]).
#[cube]
fn row_major(#[comptime] space: Space) -> (CoordsDyn, CoordsDyn) {
    let rank = comptime!(space.rank());
    let mut shape = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        shape.push(u32::from_int(comptime!(space.extent(space.axis_at(p)) as i64)));
    }
    let mut strides = CoordsDyn::new();
    #[unroll]
    for p in 0..rank {
        let mut weight = u32::from_int(1);
        #[unroll]
        for q in comptime!(p + 1)..rank {
            weight *= shape[q];
        }
        strides.push(weight);
    }
    (shape, strides)
}

/// In-kernel twin of cubecl's `TiledViewLayout` (which has no in-kernel
/// constructor): logical [`CoordsDyn`] → flat buffer offset, by splitting each
/// logical axis into its `[grid…, tile…]` parts ([`TiledLayout`]) then dotting the
/// physical strides. Built from a [`MemData`]'s stored shape/strides in
/// [`view`](Tile::view).
#[derive(CubeType, Clone)]
pub struct GmemLayout {
    physical_shape: CoordsDyn,
    physical_strides: CoordsDyn,
    #[cube(comptime)]
    start_axis: usize,
    #[cube(comptime)]
    num_tiled: usize,
    #[cube(comptime)]
    levels: usize,
}

#[cube]
impl Layout for GmemLayout {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let split = TiledLayout::new(
            self.physical_shape.clone(),
            comptime!(self.start_axis),
            self.num_tiled,
            self.levels,
        );
        let physical = split.to_source_pos(pos);
        let mut offset = u32::from_int(0);
        #[unroll]
        for i in 0..self.physical_strides.len() {
            offset += physical[i] * self.physical_strides[i];
        }
        offset as usize
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        let split = TiledLayout::new(
            self.physical_shape.clone(),
            comptime!(self.start_axis),
            self.num_tiled,
            self.levels,
        );
        split.shape()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let bounds = self.shape();
        let mut valid = true;
        #[unroll]
        for i in 0..bounds.len() {
            valid = valid && pos[i] < bounds[i];
        }
        valid
    }
}

/// The layout [`Tile::at`] applies: shift every axis to `origin` and crop it
/// to `extent`, so the sub-tile's coordinate `pos` reads source `origin + pos`.
/// Same rank as the source (it relocates and crops, it doesn't drop axes); the
/// rank-reducing 2-D slice is [`MatrixWindow`](super::MatrixWindow).
#[derive(CubeType, Clone)]
pub struct Window {
    origin: CoordsDyn,
    extent: CoordsDyn,
}

#[cube]
impl Window {
    pub fn new(origin: CoordsDyn, extent: CoordsDyn) -> Self {
        Window { origin, extent }
    }
}

#[cube]
impl Layout for Window {
    type Coordinates = CoordsDyn;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let mut out = CoordsDyn::new();
        #[unroll]
        for i in 0..self.origin.len() {
            out.push(self.origin[i] + pos[i]);
        }
        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos.clone());
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.extent.clone()
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let mut valid = true;
        #[unroll]
        for i in 0..self.extent.len() {
            valid = valid && pos[i] < self.extent[i];
        }
        valid
    }
}
