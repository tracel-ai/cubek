//! The [`Tile`]: one operand's data, a [`Payload`] backing store plus the comptime
//! [`Space`] it projects.
use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout},
    prelude::*,
    std::tensor::{
        AsView, AsViewExpand, AsViewMut, AsViewMutExpand, View, ViewMut,
        layout::{Coords1d, CoordsDyn, Layout, LayoutExpand, tiled_view::TiledLayout},
    },
};

use super::*;

/// How a launched tensor's `[pre…, grid…, tile…]` buffer maps to the logical
/// [`Space`]. A property of the tensor, distinct from the space's partitioner.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Storage {
    pub start_axis: usize,
    pub levels: usize,
}

impl Storage {
    /// Every axis tiled, no passthrough; `levels` read off the tensor's rank.
    pub fn of(physical_rank: usize, logical_rank: usize) -> Self {
        Storage {
            start_axis: 0,
            levels: physical_rank / logical_rank - 1,
        }
    }

    pub fn passthrough(start_axis: usize, levels: usize) -> Self {
        Storage { start_axis, levels }
    }
}

/// The launchable form of a [`Tile`]: a `&Tensor` plus the comptime [`Space`] and
/// [`Storage`]. The kernel turns it into a `Tile` with [`tile`](TileArg::tile).
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
    pub fn tile(&self) -> Tile<Vector<E, Const<1>>> {
        Tile::from_tensor(
            self.tensor,
            comptime!(self.space.clone()),
            comptime!(self.storage),
        )
    }
}

/// One operand's data: the runtime [`Payload`] and the comptime [`Space`] it projects.
#[derive(CubeType)]
pub struct Tile<T: CubePrimitive> {
    pub payload: Payload<T>,
    #[cube(comptime)]
    pub space: Space,
}

/// A tile's backing store. Every variant is lifetime-free (a `Box<[T]>` or a
/// [`cmma::Matrix`]); [`view`](Tile::view) rebuilds a borrowed view on demand.
#[derive(CubeType)]
pub enum Payload<T: CubePrimitive> {
    Gmem(MemData<T>),
    Smem(MemData<T>),
    /// MMA-unit-resident, not addressable (no memory view); contraction is `cmma::execute`.
    Cmma(CmmaData<T>),
}

/// A tensor-core fragment plus its comptime config. `cmma::load` picks
/// load-vs-`load_with_layout` by `ident`, and `store`/`cast` need the layout. The
/// fragment's `m`/`n`/`k` and the slice stride come from the tile's [`Space`].
#[derive(CubeType)]
pub struct CmmaData<T: CubePrimitive> {
    pub matrix: Matrix<T>,
    #[cube(comptime)]
    pub ident: MatrixIdent,
    #[cube(comptime)]
    pub layout: MatrixLayout,
}

/// The lifetime-erased buffer plus the physical shape/strides and tiling spec to
/// rebuild its [`GmemLayout`]. Fixed at construction, never recomputed from the
/// `Space`, so a staged smem sub-tile keeps addressing its whole buffer after
/// [`at`](Tile::at) windows it down.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct MemData<T: CubePrimitive> {
    buffer: Box<[T]>,
    physical_shape: CoordsDyn,
    physical_strides: CoordsDyn,
    /// Accumulates across [`at`](Tile::at)s.
    origin: CoordsDyn,
    extent: CoordsDyn,
    #[cube(comptime)]
    start_axis: usize,
    /// Tiled axes, each split into `levels + 1` `[grid…, tile…]` parts.
    #[cube(comptime)]
    num_tiled: usize,
    /// `0` = smem / untiled.
    #[cube(comptime)]
    levels: usize,
}

#[cube]
impl<T: CubePrimitive> Tile<T> {
    /// Wrap a launched [`Tensor`] into a whole `Gmem` tile. The borrow is erased
    /// into a `Box`.
    pub fn from_tensor(
        tensor: &Tensor<T>,
        #[comptime] space: Space,
        #[comptime] storage: Storage,
    ) -> Tile<T> {
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
        Tile::<T> {
            payload: Payload::new_Gmem(MemData::<T> {
                buffer,
                physical_shape,
                physical_strides,
                origin,
                extent,
                start_axis,
                num_tiled,
                levels,
            }),
            space: comptime!(space),
        }
    }

    /// Wrap a shared-memory buffer as a whole `Smem` tile. Row-major over `space`;
    /// the borrow is erased into a `Box`.
    pub fn smem(smem: &Shared<[T]>, #[comptime] space: Space) -> Tile<T> {
        let buffer = unsafe { smem.inner_ref().as_boxed_unchecked() };
        let (physical_shape, physical_strides) = row_major(comptime!(space.clone()));
        let (origin, extent) = full_window(comptime!(space.clone()));
        Tile::<T> {
            payload: Payload::new_Smem(MemData::<T> {
                buffer,
                physical_shape,
                physical_strides,
                origin,
                extent,
                start_axis: comptime!(0usize),
                num_tiled: comptime!(space.rank()),
                levels: comptime!(0usize),
            }),
            space: comptime!(space),
        }
    }

    /// Allocate an uninitialized tensor-core fragment as a `Cmma` tile. `m`/`n`/`k`
    /// are the whole MMA tile, passed in full whatever the role.
    pub fn cmma_fragment(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
        #[comptime] layout: MatrixLayout,
        #[comptime] space: Space,
    ) -> Tile<T> {
        let matrix = unsafe { Matrix::<T>::uninitialized(ident, m, n, k, layout) };
        Tile::<T> {
            payload: Payload::new_Cmma(CmmaData::<T> {
                matrix,
                ident,
                layout,
            }),
            space: comptime!(space),
        }
    }

    /// A read [`View`]: the buffer re-viewed through its base layout, then the
    /// [`Window`].
    pub fn view(&self) -> View<'_, T, CoordsDyn> {
        match &self.payload {
            Payload::Gmem(g) => g.buffer.view(g.base()).view(g.window()),
            Payload::Smem(g) => g.buffer.view(g.base()).view(g.window()),
            Payload::Cmma(_) => panic!("Tile::view: a cmma fragment has no memory view"),
        }
    }

    pub fn view_mut(&mut self) -> ViewMut<'_, T, CoordsDyn> {
        match &mut self.payload {
            Payload::Gmem(g) => {
                let base = g.base();
                let window = g.window();
                g.buffer.view_mut(base).view_mut(window)
            }
            Payload::Smem(g) => {
                let base = g.base();
                let window = g.window();
                g.buffer.view_mut(base).view_mut(window)
            }
            Payload::Cmma(_) => panic!("Tile::view_mut: a cmma fragment has no memory view"),
        }
    }

    /// Window this tile down to `region` (no copy). The tile projects `region` onto
    /// its own axes, so `lhs ∈ {M,K}` and `out ∈ {M,N}` agree without the caller
    /// matching them.
    pub fn at(&self, region: &Region) -> Tile<T> {
        let payload = match &self.payload {
            Payload::Gmem(g) => Payload::new_Gmem(g.at(region, comptime!(self.space.clone()))),
            Payload::Smem(g) => Payload::new_Smem(g.at(region, comptime!(self.space.clone()))),
            Payload::Cmma(_) => panic!("Tile::at: a cmma fragment cannot be located by view"),
        };
        Tile::<T> {
            payload,
            space: comptime!(self.space.divide()),
        }
    }

    /// Transit `src` into `self` across a level. A fragment goes through cmma
    /// load/store, memory to memory is an element copy. Moves data (unlike
    /// [`at`](Tile::at)); sync after.
    pub fn stage(&mut self, src: &Tile<T>) {
        // Read both payload kinds first, then branch, to avoid nesting a self-method
        // call inside a payload borrow.
        // `matches!` isn't supported inside `#[cube]`, so spell out the match.
        #[allow(clippy::match_like_matches_macro)]
        let frag_dst = match &self.payload {
            Payload::Cmma(_) => true,
            _ => false,
        };
        #[allow(clippy::match_like_matches_macro)]
        let frag_src = match &src.payload {
            Payload::Cmma(_) => true,
            _ => false,
        };
        if frag_dst {
            self.cmma_load(src);
        } else if frag_src {
            self.cmma_store(src);
        } else {
            self.stage_from_memory(src);
        }
    }

    /// Fill this fragment from `src`'s memory buffer: `A`/`B` use `cmma::load`, an
    /// `Accumulator` uses `load_with_layout`. The stride is the matrix row width
    /// (last-axis extent) from the space.
    fn cmma_load(&mut self, src: &Tile<T>) {
        let stride = comptime!(self.space.extent(self.space.axis_at(self.space.rank() - 1)) as u32);
        match &mut self.payload {
            Payload::Cmma(d) => match &src.payload {
                Payload::Gmem(s) => match comptime!(d.ident) {
                    MatrixIdent::Accumulator => cmma::load_with_layout(
                        &mut d.matrix,
                        &s.buffer,
                        stride,
                        comptime!(d.layout),
                    ),
                    _ => cmma::load(&mut d.matrix, &s.buffer, stride),
                },
                Payload::Smem(s) => match comptime!(d.ident) {
                    MatrixIdent::Accumulator => cmma::load_with_layout(
                        &mut d.matrix,
                        &s.buffer,
                        stride,
                        comptime!(d.layout),
                    ),
                    _ => cmma::load(&mut d.matrix, &s.buffer, stride),
                },
                Payload::Cmma(_) => panic!("Tile::stage: cmma→cmma cast not yet wired"),
            },
            // Unreachable: `stage` routes here only when `self` is a fragment.
            Payload::Gmem(_) => (),
            Payload::Smem(_) => (),
        }
    }

    /// Drain `src` (a `Cmma` fragment) into this memory tile's buffer. Stride is the
    /// matrix row width from the space.
    fn cmma_store(&mut self, src: &Tile<T>) {
        let stride = comptime!(self.space.extent(self.space.axis_at(self.space.rank() - 1)) as u32);
        match &src.payload {
            Payload::Cmma(s) => match &mut self.payload {
                Payload::Gmem(d) => {
                    cmma::store(&mut d.buffer, &s.matrix, stride, comptime!(s.layout))
                }
                Payload::Smem(d) => {
                    cmma::store(&mut d.buffer, &s.matrix, stride, comptime!(s.layout))
                }
                // Unreachable: `stage` routes here only when `self` is memory.
                Payload::Cmma(_) => (),
            },
            // Unreachable: `stage` routes here only when `src` is a fragment.
            Payload::Gmem(_) => (),
            Payload::Smem(_) => (),
        }
    }

    /// Memory to memory transit: copy each 2-D matrix of `src` into `self`
    /// element-wise.
    fn stage_from_memory(&mut self, src: &Tile<T>) {
        let matrices = self.matrix_count();
        for j in 0..matrices {
            let s = src.matrix(j);
            let mut d = self.matrix_mut(j);
            copy_2d::<T>(&mut d, &s);
        }
    }
}

#[cube]
impl<T: CubePrimitive> MemData<T> {
    /// The base layout: the `[grid…, tile…]` split (gmem, `levels > 0`) or a plain
    /// strided dot (smem, `levels = 0`).
    fn base(&self) -> GmemLayout {
        GmemLayout {
            physical_shape: self.physical_shape.clone(),
            physical_strides: self.physical_strides.clone(),
            start_axis: self.start_axis,
            num_tiled: self.num_tiled,
            levels: self.levels,
        }
    }

    fn window(&self) -> Window {
        Window::new(self.origin.clone(), self.extent.clone())
    }

    /// Window down to `region`: shift the origin by the region's tile coordinate
    /// times the sub-tile edge, crop each axis to that edge, re-box the same buffer.
    fn at(&self, region: &Region, #[comptime] space: Space) -> MemData<T> {
        let mut origin = CoordsDyn::new();
        let mut extent = CoordsDyn::new();

        #[unroll]
        for p in 0..space.rank() {
            let axis = space.axis_at(p);
            let edge = space.partitioner().edge(axis);
            let index = region.coord(axis);

            origin.push(self.origin[p] + (index * edge) as u32);
            extent.push(edge as u32);
        }

        MemData::<T> {
            buffer: unsafe { self.buffer.as_boxed_unchecked() },
            physical_shape: self.physical_shape.clone(),
            physical_strides: self.physical_strides.clone(),
            origin,
            extent,
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
    for p in 0..space.rank() {
        origin.push(0);
        extent.push(space.extent(space.axis_at(p)) as u32);
    }

    (origin, extent)
}

/// Row-major physical shape/strides over `space`'s per-axis extents, stored in the
/// smem [`MemData`] so it survives `at`'s space division.
#[cube]
fn row_major(#[comptime] space: Space) -> (CoordsDyn, CoordsDyn) {
    let rank = space.rank();
    let mut shape = CoordsDyn::new();

    #[unroll]
    for p in 0..rank {
        shape.push(space.extent(space.axis_at(p)) as u32);
    }

    let mut strides = CoordsDyn::new();

    #[unroll]
    for p in 0..rank {
        let mut weight = 1;

        #[unroll]
        for q in p + 1..rank {
            weight *= shape[q];
        }

        strides.push(weight);
    }

    (shape, strides)
}

/// In-kernel twin of cubecl's `TiledViewLayout`, which has no in-kernel
/// constructor. Splits each logical axis into its `[grid…, tile…]` parts
/// ([`TiledLayout`]) then dots the physical strides.
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
            self.start_axis,
            self.num_tiled,
            self.levels,
        );

        let physical = split.to_source_pos(pos);

        let mut offset = 0;

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
            self.start_axis,
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

/// The layout [`Tile::at`] applies: shift every axis to `origin` and crop it to
/// `extent`. Same rank as the source; the rank-reducing 2-D slice is
/// [`BatchMatrix`](super::BatchMatrix).
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
