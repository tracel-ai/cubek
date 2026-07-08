//! The [`Tile`]: one operand's data, a [`TileKind`] backing store plus the comptime
//! [`Space`] it projects. This module holds the tile-level surface — the launch arg, the
//! kind/class enums, and the operations that dispatch on the kind; each backing store's
//! own data and leaves live in its file ([`mem`], [`cmma`], [`tma`]).

mod cmma;
mod mem;
mod tma;

pub use cmma::*;
pub use mem::*;
pub use tma::*;

use cubecl::{prelude::barrier::Barrier, prelude::*};

use crate::*;

/// A tile's comptime storage class — what dispatch keys on when only the *kind* of
/// store matters, not its data. Read via [`Tile::class`](crate::Tile::class).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileClass {
    /// Addressable memory (gmem or smem).
    Mem,
    /// A single mma fragment.
    Cmma,
    /// A resident partition of mma fragments.
    CmmaPartition,
    /// A TMA tensor-map source.
    Tma,
}

/// A tile's backing store. Every variant is lifetime-free (a `Box<[T]>` or a
/// [`cmma::Matrix`](cubecl::cmma::Matrix)); [`view`](Tile::view) rebuilds a borrowed view on
/// demand.
#[derive(CubeType)]
pub enum TileKind<T: Numeric> {
    Gmem(MemData<T>),
    Smem(MemData<T>),
    /// MMA-unit-resident, not addressable (no memory view); contraction is `cmma::execute`.
    Cmma(CmmaData<T>),
    /// An instance's resident accumulator partition: `m_tiles × n_tiles` cmma fragments,
    /// comptime-indexed. Built only by [`mma_resident`](crate::matmul); contraction is the
    /// partition microkernel.
    CmmaPartition(CmmaPartition<T>),
    /// A TMA tensor-map source: not element-addressable, can only be the source of a
    /// [`stage_from`](Tile::stage_from) into shared memory, which lowers to a hardware bulk copy.
    /// Built but dormant — no launch-side constructor wires it yet (see [`Tile::from_tensor_map`]).
    TmaGmem(TmaData<T>),
}

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

#[cube]
impl<T: Numeric> Tile<T> {
    /// This tile's comptime storage [`TileClass`] — the one classifier dispatch keys on.
    pub fn class(&self) -> comptime_type!(TileClass) {
        match &self.tile_kind {
            TileKind::Gmem(_) | TileKind::Smem(_) => TileClass::Mem,
            TileKind::Cmma(_) => TileClass::Cmma,
            TileKind::CmmaPartition(_) => TileClass::CmmaPartition,
            TileKind::TmaGmem(_) => TileClass::Tma,
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
            TileKind::Cmma(_) | TileKind::CmmaPartition(_) => comptime!(1usize),
            TileKind::TmaGmem(_) => comptime!(1usize),
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
            // A resident fragment (or partition of them) passes through unchanged:
            // [`mma_resident`](crate::matmul) guarantees one partition per instance, so
            // every region an instance visits selects these same fragments.
            TileKind::Cmma(c) => TileKind::new_Cmma(c.clone()),
            TileKind::CmmaPartition(p) => TileKind::new_CmmaPartition(p.clone()),
        };
        Tile::<T> {
            tile_kind,
            space: comptime!(self.space.divide()),
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
            TileKind::Cmma(_) | TileKind::CmmaPartition(_) => {
                panic!("Tile::runtime_extent: a cmma fragment has no extent")
            }
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
}
