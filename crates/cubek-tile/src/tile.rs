//! The [`Tile`]: one operand's data, a [`TileKind`] backing store plus the comptime
//! [`Space`] it projects.
use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout},
    prelude::barrier::Barrier,
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
    /// Whether this operand's logical extent can overhang its tile grid, so edge
    /// reads/writes must be bounds-checked. Set from divisibility at launch; `false`
    /// keeps the unchecked (divisible) fast path.
    pub check_bounds: bool,
}

impl Storage {
    /// Every axis tiled, no passthrough; `levels` read off the tensor's rank.
    pub fn of(physical_rank: usize, logical_rank: usize) -> Self {
        Storage {
            start_axis: 0,
            levels: physical_rank / logical_rank - 1,
            check_bounds: false,
        }
    }

    pub fn passthrough(start_axis: usize, levels: usize) -> Self {
        Storage {
            start_axis,
            levels,
            check_bounds: false,
        }
    }

    /// Set whether edge reads/writes must be bounds-checked.
    pub fn checked(mut self, check_bounds: bool) -> Self {
        self.check_bounds = check_bounds;
        self
    }
}

/// The launchable form of a [`Tile`]: a scalar `&Tensor` plus its comptime line
/// [`vector_size`](Self::vector_size), [`Space`] and [`Storage`]. The kernel turns it into a `Tile`
/// with [`tile`](TileArg::tile). The physical vectorization is a plain comptime value (the
/// `vector_size` field), not a type parameter — the buffer is served scalar and re-grouped into
/// `Vector<E, vector_size>` lines in-kernel.
#[derive(CubeType, CubeLaunch)]
pub struct TileArg<'a, E: Numeric> {
    pub tensor: &'a Tensor<E>,
    /// Physical vectorization (`Vector<E, vector_size>` line size) of the operand's contiguous
    /// innermost axis; `1` is scalar.
    #[cube(comptime)]
    pub vector_size: usize,
    #[cube(comptime)]
    pub space: Space,
    #[cube(comptime)]
    pub storage: Storage,
}

#[cube]
impl<'a, E: Numeric> TileArg<'a, E> {
    pub fn tile(&self) -> Tile<E> {
        Tile::from_tensor(
            self.tensor,
            comptime!(self.vector_size),
            comptime!(self.space.clone()),
            comptime!(self.storage),
        )
    }
}

/// One operand's data: the runtime [`TileKind`] and the comptime [`Space`] it projects. The
/// generic `T` is the element the tile *serves/computes* in; the physical vectorization is a
/// storage detail held inside the [`TileKind`] variant (read via [`vector_size`](Tile::vector_size)).
#[derive(CubeType)]
pub struct Tile<T: Numeric> {
    pub tile_kind: TileKind<T>,
    #[cube(comptime)]
    pub space: Space,
}

/// A tensor-core fragment plus its comptime config. `cmma::load` picks
/// load-vs-`load_with_layout` by `ident`, and `store`/`cast` need the layout. The
/// fragment's `m`/`n`/`k` and the slice stride come from the tile's [`Space`].
#[derive(CubeType)]
pub struct CmmaData<T: Numeric> {
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
pub struct MemData<T: Numeric> {
    /// Scalar-typed backing store. Its physical vectorization is erased from the type and held in
    /// [`vector_size`](MemData::vector_size); buffer access re-groups it into `Vector<T, W>` lines
    /// (the width passed by the caller) so the line-unit strides/extents still address it.
    buffer: Box<[T]>,
    /// Physical vectorization (`Vector<T, vector_size>` line size) of the backing store: the launched
    /// operand's vector size, `1` for an unvectorized store. The leaf reconstructs `Vector<T, W>`
    /// from it; held comptime so `size!` can read it.
    #[cube(comptime)]
    vector_size: usize,
    physical_shape: CoordsDyn,
    physical_strides: CoordsDyn,
    /// Accumulates across [`at`](Tile::at)s.
    origin: CoordsDyn,
    extent: CoordsDyn,
    /// Absolute logical extent per axis (the valid region). `origin + pos` beyond this is
    /// the partial-tile overhang. Preserved across [`at`](Tile::at), unlike `extent`
    /// (the tile cell size). Runtime, so one kernel serves any shape.
    bound: CoordsDyn,
    #[cube(comptime)]
    start_axis: usize,
    /// Tiled axes, each split into `levels + 1` `[grid…, tile…]` parts.
    #[cube(comptime)]
    num_tiled: usize,
    /// `0` = smem / untiled.
    #[cube(comptime)]
    levels: usize,
    /// Whether edge reads/writes must be bounds-checked (the logical extent overhangs
    /// the tile grid). `false` is the unchecked fast path. Smem never overhangs, so it's
    /// always `false` there; gmem inherits its operand's launch-time flag.
    #[cube(comptime)]
    check: bool,
}

/// A TMA tensor-map source: the launch-built `ViewMut` (backed by a `TensorMapTiled` `ViewArg`),
/// the current global box origin `pos`, the logical `bound`, and the comptime box shape. Not
/// element-addressable — its only sink is a [`stage_from`](Tile::stage_from) into shared memory, which
/// lowers to a hardware bulk copy. `at` advances `pos`; the descriptor and bound ride along unchanged.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct TmaData<T: Numeric> {
    view: ViewMut<'static, T, CoordsDyn>,
    pos: CoordsDyn,
    bound: CoordsDyn,
    #[cube(comptime)]
    box_rows: u32,
    #[cube(comptime)]
    box_cols: u32,
    #[cube(comptime)]
    transposed: bool,
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// Wrap a launched scalar [`Tensor`] into a whole `Gmem` tile. The borrow is erased into a `Box`
    /// and `vector_size` is recorded as the store's [`vector_size`](Tile::vector_size). Strides stay
    /// scalar-unit (the leaf addresses the buffer with scalar offsets); only the contiguous innermost
    /// axis re-expresses as `Vector<T, vector_size>` lines — its extent shrinks to a line count
    /// (`extent / w`) and its step grows to `w` scalars (`stride * w`). All a no-op at
    /// `vector_size == 1`.
    pub fn from_tensor(
        tensor: &Tensor<T>,
        #[comptime] vector_size: usize,
        #[comptime] space: Space,
        #[comptime] storage: Storage,
    ) -> Tile<T> {
        let start_axis = comptime!(storage.start_axis);
        let num_tiled = comptime!(space.rank() - storage.start_axis);
        let levels = comptime!(storage.levels);
        let rank = comptime!(start_axis + (levels + 1) * num_tiled);
        let last = comptime!(rank - 1);
        let w = comptime!(vector_size as u32);
        let mut physical_shape = CoordsDyn::new();
        let mut physical_strides = CoordsDyn::new();
        #[unroll]
        for i in 0..rank {
            let extent = tensor.shape(i) as u32;
            let stride = tensor.stride(i) as u32;
            if comptime!(i == last) {
                // Innermost (contiguous, stride 1): count lines (extent / w), each a `w`-scalar
                // step (stride * w). The offset stays scalar-unit; the leaf loads a `w`-wide line.
                physical_shape.push(extent / w);
                physical_strides.push(stride * w);
            } else {
                // Coarser axes keep their scalar strides.
                physical_shape.push(extent);
                physical_strides.push(stride);
            }
        }
        let buffer = unsafe { tensor.as_slice().as_boxed_unchecked() };
        // Logical bound folded from the physical shape, so it's correct for tiled
        // operands too (the physical buffer is padded; the logical extent is not).
        let bound = logical_bound(&physical_shape, start_axis, num_tiled, levels);
        // The whole-tile window. A `Dynamic` axis takes its runtime size from `bound`, so the
        // top-level extent never bakes into the kernel; a `Static` axis keeps its comptime size.
        let (origin, extent) = top_window(comptime!(space.clone()), &bound, vector_size);
        Tile::<T> {
            tile_kind: TileKind::new_Gmem(MemData::<T> {
                buffer,
                vector_size: comptime!(vector_size),
                physical_shape,
                physical_strides,
                origin,
                extent,
                bound,
                start_axis,
                num_tiled,
                levels,
                check: comptime!(storage.check_bounds),
            }),
            space: comptime!(space),
        }
    }

    /// Allocate a fresh shared-memory tile shaped to stage one `divide()` sub-tile of `self`, at the
    /// same physical width. The one-liner staging schedules reach for — `lhs.smem_like()` instead of
    /// hand-rolling a `Shared` slice, its size, and a `Tile::smem`.
    pub fn smem_like(&self) -> Tile<T> {
        Tile::smem(comptime!(self.space.divide()), self.vector_size())
    }

    /// Allocate a shared-memory tile row-major over `space`, at physical `vector_size`. The scalar
    /// slice (`space.tile_size() * vector_size` entries) is allocated here and erased into a `Box`.
    pub fn smem(#[comptime] space: Space, #[comptime] vector_size: usize) -> Tile<T> {
        let smem = Shared::<[T]>::new_slice(space.tile_size() * vector_size);
        let buffer = unsafe { smem.inner_ref().as_boxed_unchecked() };
        let (physical_shape, physical_strides) = row_major(comptime!(space.clone()), vector_size);
        let (origin, extent) = full_window(comptime!(space.clone()), vector_size);
        // Smem is its own full buffer — never overhangs — so the bound is the extent and
        // checks are off.
        let bound = extent.clone();
        Tile::<T> {
            tile_kind: TileKind::new_Smem(MemData::<T> {
                buffer,
                vector_size,
                physical_shape,
                physical_strides,
                origin,
                extent,
                bound,
                start_axis: comptime!(0usize),
                num_tiled: comptime!(space.rank()),
                levels: comptime!(0usize),
                check: comptime!(false),
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
            tile_kind: TileKind::new_Cmma(CmmaData::<T> {
                matrix,
                ident,
                layout,
            }),
            space: comptime!(space),
        }
    }

    /// Wrap a TMA tensor-map [`ViewMut`] (built on the client side) as a `TmaGmem` tile. The global
    /// `(batch, row, col)` bound is read off the view; `pos` starts at the origin and advances on
    /// [`at`](Tile::at). Element addressing is unavailable — its only sink is a
    /// [`stage_from`](Tile::stage_from) into shared memory. The store is scalar (`vector_size == 1`); the box
    /// shape is carried comptime for the `tensor_map_load`. Dormant: no launch path builds this yet.
    pub fn from_tensor_map(
        view: ViewMut<'static, T, CoordsDyn>,
        #[comptime] space: Space,
        #[comptime] box_rows: u32,
        #[comptime] box_cols: u32,
        #[comptime] transposed: bool,
    ) -> Tile<T> {
        let bound = view.shape();
        let mut pos = CoordsDyn::new();
        #[unroll]
        for _ in 0..comptime!(space.rank()) {
            pos.push(0u32);
        }
        Tile::<T> {
            tile_kind: TileKind::new_TmaGmem(TmaData::<T> {
                view,
                pos,
                bound,
                box_rows,
                box_cols,
                transposed,
            }),
            space: comptime!(space),
        }
    }

    /// This operand's runtime logical size along `axis`, read off the [`bound`](MemData) folded
    /// from the tensor shape. The source of a [`Dynamic`](crate::Extent) axis's tile count, so
    /// one kernel serves any shape. A cmma fragment has no buffer extent.
    pub fn runtime_extent(&self, #[comptime] axis: Axis) -> usize {
        let p = comptime!(self.space.position(axis));
        let raw = match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => g.bound[p] as usize,
            TileKind::TmaGmem(t) => t.bound[p] as usize,
            TileKind::Cmma(_) => panic!("Tile::runtime_extent: a cmma fragment has no extent"),
        };
        // `bound` is a line count on the vectorized innermost axis (folded from the lined physical
        // shape); the walk divides by conceptual edges, so return line count × width. No-op off the
        // innermost axis and at width 1.
        let last = comptime!(self.space.rank() - 1);
        let w = self.vector_size();
        comptime!(if p == last { w } else { 1usize }) * raw
    }

    /// The runtime space to walk this tile: its comptime tiling spec plus the runtime sizes of any
    /// `Dynamic` axes, read off the tile. A fully-`Static` tile short-circuits to no runtime sizes.
    pub fn runtime_space(&self) -> Space {
        let space = comptime!(self.space.clone());
        let mut sizes = Sequence::<usize>::new();
        if comptime!(!space.is_static()) {
            #[unroll]
            for p in 0..comptime!(space.rank()) {
                sizes.push(self.runtime_extent(space.axis_at(p)));
            }
        }
        Space::with_sizes(space, sizes)
    }

    /// A read [`View`] over `Vector<T, W>` lines: the scalar buffer re-grouped into its physical
    /// width, then re-viewed through the base layout and [`Window`]. `W` is the line width
    /// (`self.vector_size`); pass `Const<1>` when only the (width-invariant) leading shape is needed.
    pub fn view<W: Size>(&self) -> View<'_, Vector<T, W>, CoordsDyn> {
        match &self.tile_kind {
            TileKind::Gmem(g) => g.lines::<W>().view(g.base()).view(g.window()),
            TileKind::Smem(g) => g.lines::<W>().view(g.base()).view(g.window()),
            TileKind::TmaGmem(_) => panic!("Tile::view: a tma source has no element view"),
            TileKind::Cmma(_) => panic!("Tile::view: a cmma fragment has no memory view"),
        }
    }

    pub fn view_mut<W: Size>(&mut self) -> ViewMut<'_, Vector<T, W>, CoordsDyn> {
        match &mut self.tile_kind {
            TileKind::Gmem(g) => {
                let base = g.base();
                let window = g.window();
                g.lines_mut::<W>().view_mut(base).view_mut(window)
            }
            TileKind::Smem(g) => {
                let base = g.base();
                let window = g.window();
                g.lines_mut::<W>().view_mut(base).view_mut(window)
            }
            TileKind::TmaGmem(_) => panic!("Tile::view_mut: a tma source has no element view"),
            TileKind::Cmma(_) => panic!("Tile::view_mut: a cmma fragment has no memory view"),
        }
    }

    /// Window this tile down to `region` (no copy). The tile projects `region` onto
    /// its own axes, so `lhs ∈ {M,K}` and `out ∈ {M,N}` agree without the caller
    /// matching them.
    pub fn at(&self, region: &Region) -> Tile<T> {
        let tile_kind = match &self.tile_kind {
            TileKind::Gmem(g) => TileKind::new_Gmem(g.at(region, comptime!(self.space.clone()))),
            TileKind::Smem(g) => TileKind::new_Smem(g.at(region, comptime!(self.space.clone()))),
            TileKind::TmaGmem(t) => {
                TileKind::new_TmaGmem(t.at(region, comptime!(self.space.clone())))
            }
            TileKind::Cmma(_) => panic!("Tile::at: a cmma fragment cannot be located by view"),
        };
        Tile::<T> {
            tile_kind,
            space: comptime!(self.space.divide()),
        }
    }

    /// Whether this operand is delivered by TMA (an async hardware bulk-copy source) rather than a
    /// strided element copy. Comptime (the tile kind is fixed at trace); drives the staging sync.
    #[allow(clippy::match_like_matches_macro)] // `matches!` isn't supported inside `#[cube]`.
    pub fn is_tma(&self) -> comptime_type!(bool) {
        match &self.tile_kind {
            TileKind::TmaGmem(_) => true,
            _ => false,
        }
    }

    /// Physical vectorization of the backing store — the `Vector<T, vector_size>` line width the leaf
    /// reconstructs. A launched memory tile carries its operand's vector size; a cmma fragment and a
    /// tma source are scalar (`1`). Comptime; a storage detail, not part of the logical `Space`.
    pub fn vector_size(&self) -> comptime_type!(usize) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.vector_size,
            TileKind::Cmma(_) => comptime!(1usize),
            TileKind::TmaGmem(_) => comptime!(1usize),
        }
    }

    /// Blocking copy of `src` into `self` across a level, dispatched by the two tiles' kinds to a
    /// transport leaf: a fragment goes through cmma [`load`](Tile::cmma_load)/[`store`](Tile::cmma_store),
    /// a TMA source through a self-contained bulk copy ([`tma_load`](Tile::tma_load)), memory to memory
    /// is an element [`copy`](Tile::mem_copy). The pipelined (barrier-hoisted) counterpart is
    /// [`stage_from`](Tile::stage_from). Moves data (unlike [`at`](Tile::at)); returns once the data has landed.
    pub fn copy_from(&mut self, src: &Tile<T>) {
        // Read both tile-kind variants first, then branch, to avoid nesting a self-method
        // call inside a tile_kind borrow.
        // `matches!` isn't supported inside `#[cube]`, so spell out the match.
        #[allow(clippy::match_like_matches_macro)]
        let frag_dst = match &self.tile_kind {
            TileKind::Cmma(_) => true,
            _ => false,
        };
        #[allow(clippy::match_like_matches_macro)]
        let frag_src = match &src.tile_kind {
            TileKind::Cmma(_) => true,
            _ => false,
        };
        #[allow(clippy::match_like_matches_macro)]
        let tma_src = match &src.tile_kind {
            TileKind::TmaGmem(_) => true,
            _ => false,
        };
        if frag_dst {
            self.cmma_load(src);
        } else if frag_src {
            self.cmma_store(src);
        } else if tma_src {
            self.tma_load(src);
        } else {
            self.mem_copy(src);
        }
    }

    /// Bytes this shared-memory tile's buffer spans — the transaction count a TMA fill lands.
    /// Runtime (the buffer length). Zero for the non-smem kinds, which are never TMA fill targets.
    fn buffer_bytes(&self) -> u32 {
        match &self.tile_kind {
            TileKind::Smem(d) => d.buffer.len() as u32 * comptime!(T::type_size() as u32),
            TileKind::Gmem(_) => 0,
            TileKind::Cmma(_) => 0,
            TileKind::TmaGmem(_) => 0,
        }
    }

    /// Pipelined (barrier-hoisted) copy of `src` into this staged tile under `barrier`, the
    /// double-buffered counterpart of [`copy_from`](Tile::copy_from). The barrier sequences producer vs
    /// consumer; how the fill moves the bytes is read off the source, so the caller passes no flag. A TMA
    /// source declares its transaction bytes (`expect_tx`, elected unit) and pushes an async bulk copy
    /// onto `barrier` ([`tma_stage`](Tile::tma_stage), wait hoisted to the consumer); any other source is
    /// a plain synchronous element [`copy_from`](Tile::copy_from) — TMA is just one way to fill under a
    /// barrier.
    pub fn stage_from(&mut self, src: &Tile<T>, barrier: &Shared<Barrier>) {
        if src.is_tma() {
            if UNIT_POS == 0 {
                barrier.expect_tx(self.buffer_bytes());
            }
            self.tma_stage(src, barrier);
        } else {
            self.copy_from(src);
        }
    }

    /// TMA transport leaf, pipelined: issue the elected `tensor_map_load` of `src` (a tensor-map source)
    /// into this shared-memory tile onto `barrier`, without arriving or waiting — the caller hoists those
    /// so the copy overlaps compute. The core shared by [`stage_from`](Tile::stage_from) (barrier hoisted) and the
    /// blocking [`tma_load`](Tile::tma_load) (barrier owned locally).
    fn tma_stage(&mut self, src: &Tile<T>, barrier: &Shared<Barrier>) {
        match &src.tile_kind {
            TileKind::TmaGmem(s) => match &mut self.tile_kind {
                TileKind::Smem(d) => {
                    // The bulk copy is issued by one elected unit only; the transaction count
                    // declared by the caller is that unit's alone, so more issuers would over-count
                    // and corrupt the stage.
                    if UNIT_POS == 0 {
                        s.view
                            .tensor_map_load(barrier, d.buffer.downcast_mut(), s.pos.clone());
                    }
                }
                // TMA only ever targets shared memory; the other sinks are unreachable.
                TileKind::Gmem(_) => (),
                TileKind::Cmma(_) => (),
                TileKind::TmaGmem(_) => (),
            },
            // Unreachable: `stage`/`tma_load` route here only when `src` is a tma source.
            TileKind::Gmem(_) => (),
            TileKind::Smem(_) => (),
            TileKind::Cmma(_) => (),
        }
    }

    /// TMA transport leaf, blocking: hardware bulk-copy `src` (a tensor-map source) into this
    /// shared-memory tile and wait for it. Owns its `mbarrier` locally — arms it, issues the copy via
    /// [`tma_stage`](Tile::tma_stage), then `arrive_and_expect_tx` + `wait` before returning. The
    /// double-buffered path hoists that barrier out with [`stage_from`](Tile::stage_from) instead.
    fn tma_load(&mut self, src: &Tile<T>) {
        let barrier = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
        sync_async_proxy_shared();
        // One elected issuer only, matching the declared transaction count; more issuers over-count.
        let expected = select(UNIT_POS == 0, self.buffer_bytes(), 0);
        self.tma_stage(src, &barrier);
        let token = barrier.arrive_and_expect_tx(1, expected);
        barrier.wait(token);
    }

    /// Fill this fragment from `src`'s memory buffer: `A`/`B` use `cmma::load`, an
    /// `Accumulator` uses `load_with_layout`. The stride is the matrix row width
    /// (last-axis extent) from the space.
    fn cmma_load(&mut self, src: &Tile<T>) {
        let stride = comptime!(self.space.extent(self.space.axis_at(self.space.rank() - 1)) as u32);
        match &mut self.tile_kind {
            TileKind::Cmma(d) => match &src.tile_kind {
                TileKind::Gmem(s) => match comptime!(d.ident) {
                    MatrixIdent::Accumulator => cmma::load_with_layout(
                        &mut d.matrix,
                        &s.buffer,
                        stride,
                        comptime!(d.layout),
                    ),
                    _ => cmma::load(&mut d.matrix, &s.buffer, stride),
                },
                TileKind::Smem(s) => match comptime!(d.ident) {
                    MatrixIdent::Accumulator => cmma::load_with_layout(
                        &mut d.matrix,
                        &s.buffer,
                        stride,
                        comptime!(d.layout),
                    ),
                    _ => cmma::load(&mut d.matrix, &s.buffer, stride),
                },
                TileKind::Cmma(_) => panic!("Tile::copy_from: cmma→cmma cast not yet wired"),
                TileKind::TmaGmem(_) => {
                    panic!("Tile::copy_from: cmma load straight from a tma source not wired")
                }
            },
            // Unreachable: `copy_from` routes here only when `self` is a fragment.
            TileKind::Gmem(_) => (),
            TileKind::Smem(_) => (),
            TileKind::TmaGmem(_) => (),
        }
    }

    /// Drain `src` (a `Cmma` fragment) into this memory tile's buffer. Stride is the
    /// matrix row width from the space.
    fn cmma_store(&mut self, src: &Tile<T>) {
        let stride = comptime!(self.space.extent(self.space.axis_at(self.space.rank() - 1)) as u32);
        match &src.tile_kind {
            TileKind::Cmma(s) => match &mut self.tile_kind {
                TileKind::Gmem(d) => {
                    cmma::store(&mut d.buffer, &s.matrix, stride, comptime!(s.layout))
                }
                TileKind::Smem(d) => {
                    cmma::store(&mut d.buffer, &s.matrix, stride, comptime!(s.layout))
                }
                // Unreachable: `copy_from` routes here only when `self` is memory.
                TileKind::Cmma(_) => (),
                TileKind::TmaGmem(_) => (),
            },
            // Unreachable: `copy_from` routes here only when `src` is a fragment.
            TileKind::Gmem(_) => (),
            TileKind::Smem(_) => (),
            TileKind::TmaGmem(_) => (),
        }
    }

    /// Memory transport leaf: copy each 2-D matrix of `src` into `self` element-wise. Both tiles
    /// share `self`'s width (smem is staged at the source operand's width), so the copy moves whole
    /// `Vector<T, W>` lines.
    fn mem_copy(&mut self, src: &Tile<T>) {
        let size!(W) = self.vector_size();
        let matrices = self.matrix_count();
        for j in 0..matrices {
            let s = src.matrix::<W>(j);
            let mut d = self.matrix_mut::<W>(j);
            copy_2d::<Vector<T, W>>(&mut d, &s);
        }
    }
}

#[cube]
impl<T: Numeric> TmaData<T> {
    /// Window down to `region`: advance the global origin by each axis's tile coordinate times its
    /// sub-tile edge. The descriptor and bound carry through unchanged — only `pos` moves — so the
    /// next `tensor_map_load` copies the windowed box.
    fn at(&self, region: &Region, #[comptime] space: Space) -> TmaData<T> {
        let mut pos = CoordsDyn::new();

        #[unroll]
        for p in 0..space.rank() {
            let axis = space.axis_at(p);
            let edge = space.partitioner().edge(axis);
            let index = region.coord(axis);
            pos.push(self.pos[p] + (index * edge) as u32);
        }

        TmaData::<T> {
            view: self.view.clone(),
            pos,
            bound: self.bound.clone(),
            box_rows: comptime!(self.box_rows),
            box_cols: comptime!(self.box_cols),
            transposed: comptime!(self.transposed),
        }
    }
}

#[cube]
impl<T: Numeric> MemData<T> {
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
        Window::new(self.origin.clone(), self.extent.clone(), self.bound.clone())
    }

    /// The scalar buffer re-grouped into `Vector<T, W>` lines, so the base/window layouts address it.
    /// `W` is the store's physical [`vector_size`](Tile::vector_size). The leaf dots the scalar-unit
    /// strides for the offset and reads a `Vector<T, W>` at that scalar position; the re-grouped view
    /// reports its length in *lines* (`scalars / W`), so an unchecked read is fine but a checked one
    /// clips against the line count — hence vectorization is gated to the no-overhang path in launch.
    fn lines<W: Size>(&self) -> &[Vector<T, W>] {
        self.buffer.as_vectorized().with_vector_size::<W>()
    }

    /// The mutable twin of [`lines`](MemData::lines).
    fn lines_mut<W: Size>(&mut self) -> &mut [Vector<T, W>] {
        self.buffer.as_vectorized_mut().with_vector_size_mut::<W>()
    }

    /// Re-view this buffer through `layout` as a [`MatrixView`], carrying its own `check` flag
    /// so the leaf masks without being asked.
    pub(crate) fn masked<W: Size>(&self, layout: BatchMatrix) -> MatrixView<'_, Vector<T, W>> {
        MaskedView::new(
            self.lines::<W>()
                .view(self.base())
                .view(self.window())
                .view(layout),
            comptime!(self.check),
        )
    }

    /// The mutable twin of [`masked`](MemData::masked).
    pub(crate) fn masked_mut<W: Size>(
        &mut self,
        layout: BatchMatrix,
    ) -> MatrixViewMut<'_, Vector<T, W>> {
        let base = self.base();
        let window = self.window();
        let check = comptime!(self.check);
        MaskedViewMut::new(
            self.lines_mut::<W>()
                .view_mut(base)
                .view_mut(window)
                .view_mut(layout),
            check,
        )
    }

    /// Re-view this buffer as a flat 1-D [`FlatView`] over its [`Window`] extent: a
    /// [`FlatLayout`] turns a row-major index into the N-D position, carrying the `check` flag
    /// so a flat scan masks the overhang without being asked.
    pub(crate) fn flat<W: Size>(&self) -> FlatView<'_, Vector<T, W>> {
        FlatView::new(
            self.lines::<W>()
                .view(self.base())
                .view(self.window())
                .view(FlatLayout::new(self.extent.clone())),
            comptime!(self.check),
        )
    }

    /// The mutable twin of [`flat`](MemData::flat).
    pub(crate) fn flat_mut<W: Size>(&mut self) -> FlatViewMut<'_, Vector<T, W>> {
        let base = self.base();
        let window = self.window();
        let extent = self.extent.clone();
        let check = comptime!(self.check);
        FlatViewMut::new(
            self.lines_mut::<W>()
                .view_mut(base)
                .view_mut(window)
                .view_mut(FlatLayout::new(extent)),
            check,
        )
    }

    /// The `i`-th batch matrix as a 2-D view. Mirrors [`Tile::matrix_mut`] for callers that
    /// hold the tile-kind rather than the whole tile, so the `space` is passed in.
    pub(crate) fn matrix_mut<W: Size>(
        &mut self,
        i: usize,
        #[comptime] space: Space,
    ) -> MatrixViewMut<'_, Vector<T, W>> {
        let rank = comptime!(space.rank());
        let rows = comptime!(space.extent_at(rank - 2));
        // The `Space` is scalar; `cols` counts lines, so divide the innermost extent by the width.
        let cols = comptime!(space.extent_at(rank - 1) / self.vector_size);
        // Leading (batch) extents are width-invariant, so a `Const<1>` regroup gives the right shape.
        let shape = self
            .lines::<Const<1>>()
            .view(self.base())
            .view(self.window())
            .shape();
        let mut batches = CoordsDyn::new();
        #[unroll]
        for p in 0..rank - 2 {
            let mut weight = 1;
            #[unroll]
            for q in comptime!(p + 1)..rank - 2 {
                weight *= shape[q];
            }
            batches.push((i as u32 / weight) % shape[p]);
        }
        self.masked_mut::<W>(BatchMatrix::new(batches, rows, cols))
    }

    /// Window down to `region`: shift the origin by the region's tile coordinate
    /// times the sub-tile edge, crop each axis to that edge, re-box the same buffer.
    /// `bound` (the absolute valid extent) is carried through unchanged — only `origin`
    /// moves — so the leaf masks `origin + pos < bound` regardless of nesting depth.
    fn at(&self, region: &Region, #[comptime] space: Space) -> MemData<T> {
        let mut origin = CoordsDyn::new();
        let mut extent = CoordsDyn::new();

        let last = comptime!(space.rank() - 1);
        #[unroll]
        for p in 0..space.rank() {
            let axis = space.axis_at(p);
            // The innermost (vectorized) axis's edge is a line count, so `/ width`.
            let edge = comptime!(if p == last {
                space.partitioner().edge(axis) / self.vector_size
            } else {
                space.partitioner().edge(axis)
            });
            let index = region.coord(axis);

            origin.push(self.origin[p] + (index * edge) as u32);
            extent.push(edge as u32);
        }

        MemData::<T> {
            buffer: unsafe { self.buffer.as_boxed_unchecked() },
            vector_size: comptime!(self.vector_size),
            physical_shape: self.physical_shape.clone(),
            physical_strides: self.physical_strides.clone(),
            origin,
            extent,
            bound: self.bound.clone(),
            start_axis: comptime!(self.start_axis),
            num_tiled: comptime!(self.num_tiled),
            levels: comptime!(self.levels),
            check: comptime!(self.check),
        }
    }
}

/// The operand's logical extent per axis, folded from its physical `[pre…, grid…, tile…]`
/// shape: passthrough axes pass through, each tiled axis multiplies its per-level factors.
/// Reduces to `physical_shape` for an untiled (strided) operand.
#[cube]
fn logical_bound(
    physical_shape: &CoordsDyn,
    #[comptime] start_axis: usize,
    #[comptime] num_tiled: usize,
    #[comptime] levels: usize,
) -> CoordsDyn {
    let mut bound = CoordsDyn::new();
    #[unroll]
    for i in 0..start_axis {
        bound.push(physical_shape[i]);
    }
    #[unroll]
    for a in 0..num_tiled {
        let mut prod = 1u32;
        #[unroll]
        for l in 0..comptime!(levels + 1) {
            prod *= physical_shape[comptime!(start_axis) + l * num_tiled + a];
        }
        bound.push(prod);
    }
    bound
}

/// The whole-tile window: `origin = 0`, `extent =` the space's per-axis extents. `Space` is
/// conceptual; the innermost (vectorized) axis's extent is a line count, `/ vector_size`.
#[cube]
fn full_window(#[comptime] space: Space, #[comptime] vector_size: usize) -> (CoordsDyn, CoordsDyn) {
    let mut origin = CoordsDyn::new();
    let mut extent = CoordsDyn::new();
    let last = comptime!(space.rank() - 1);

    #[unroll]
    for p in 0..space.rank() {
        origin.push(0);
        let e = comptime!(space.extent(space.axis_at(p)));
        extent.push(comptime!(if p == last { e / vector_size } else { e }) as u32);
    }

    (origin, extent)
}

/// [`full_window`] for the top gmem tile, where an axis may be [`Dynamic`](crate::Extent): such
/// an axis reads its runtime size from `bound` (the folded logical extent) instead of a comptime
/// constant, so the problem shape never specializes the kernel.
#[cube]
fn top_window(
    #[comptime] space: Space,
    bound: &CoordsDyn,
    #[comptime] vector_size: usize,
) -> (CoordsDyn, CoordsDyn) {
    let mut origin = CoordsDyn::new();
    let mut extent = CoordsDyn::new();
    let last = comptime!(space.rank() - 1);

    #[unroll]
    for p in 0..space.rank() {
        origin.push(0);
        let axis = comptime!(space.axis_at(p));
        // The innermost (vectorized) axis is a line count, `/ vector_size`. A `Dynamic` axis reads
        // its size from `bound`, already lined from the physical shape.
        let size = match comptime!(space.extent_raw(axis)) {
            Extent::Static(e) => {
                (comptime!(if p == last { e / vector_size } else { e }) as u32).runtime()
            }
            Extent::Dynamic => bound[p],
        };
        extent.push(size);
    }

    (origin, extent)
}

/// Row-major physical shape/strides over `space`'s per-axis extents, stored in the smem [`MemData`]
/// so it survives `at`'s space division. The innermost (vectorized) axis's extent is a line count
/// (`/ vector_size`) and strides stay scalar-unit (its line step is `vector_size` scalars); all a
/// no-op at `vector_size == 1`.
#[cube]
fn row_major(#[comptime] space: Space, #[comptime] vector_size: usize) -> (CoordsDyn, CoordsDyn) {
    let rank = space.rank();
    let last = comptime!(rank - 1);
    let mut shape = CoordsDyn::new();

    #[unroll]
    for p in 0..rank {
        let e = comptime!(space.extent(space.axis_at(p)));
        shape.push(comptime!(if p == last { e / vector_size } else { e }) as u32);
    }

    let mut strides = CoordsDyn::new();

    #[unroll]
    for p in 0..rank {
        // Weight over the *scalar* extents so coarser strides stay scalar; the innermost line step
        // widens to `vector_size`.
        let weight = comptime! {
            let mut acc = 1;
            for q in (p + 1)..rank {
                acc *= space.extent(space.axis_at(q));
            }
            acc
        };
        strides.push(comptime!(if p == last {
            weight * vector_size
        } else {
            weight
        }) as u32);
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
    /// Absolute logical extent (the valid region). `shape()` stays `extent` (the tile
    /// cell, so loops cover the whole padded tile), but `is_in_bounds` clips against
    /// `bound` so a checked read/write zeroes / skips the overhang.
    bound: CoordsDyn,
}

#[cube]
impl Window {
    pub fn new(origin: CoordsDyn, extent: CoordsDyn, bound: CoordsDyn) -> Self {
        Window {
            origin,
            extent,
            bound,
        }
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

        // The cell can overhang the matrix; a position is valid only if its absolute
        // coordinate (`origin + pos`) is within the logical `bound`.
        #[unroll]
        for i in 0..self.bound.len() {
            valid = valid && self.origin[i] + pos[i] < self.bound[i];
        }

        valid
    }
}
