//! The [`Tile`]: one operand's data, a [`TileKind`] backing store plus the comptime
//! [`Space`] it projects. Kernel-side structure only; each backing store's own data and
//! leaves live in its file ([`mem`], [`cmma`], [`tma`]), and the launch surface (the
//! arguments, deliveries and builder a tile is loaded through) lives in [`crate::load`].

mod cmma;
mod mem;
mod tma;
mod view;

pub use cmma::*;
pub use mem::*;
pub use tma::*;
pub use view::*;

use cubecl::{
    prelude::*,
    quant::scheme::{QuantLevel, QuantScheme},
};

use crate::*;

/// A tile's backing store. Every variant is lifetime-free (a `Box<[T]>` or a
/// [`cmma::Matrix`](cubecl::cmma::Matrix)); [`view`](Tile::view) rebuilds a borrowed view on
/// demand.
#[derive(CubeType)]
pub enum TileKind<T: Numeric> {
    Gmem(MemData<T>),
    Smem(MemData<T>),
    /// MMA-unit-resident, not addressable (no memory view); contraction is `cmma::execute`.
    Cmma(CmmaData<T>),
    /// A partition of cmma fragments, `m_tiles × n_tiles`, comptime-indexed; only a
    /// static walk's regions (constant coordinates) can select through it.
    CmmaPartition(CmmaPartition<T>),
    /// A TMA tensor-map source: not element-addressable, its only sink is a hardware bulk
    /// copy into shared memory. Launched via [`TmaTileArg`], the twin of [`StridedTileArg`].
    TmaGmem(TmaData<T>),
}

#[cube]
impl<T: Numeric> TileKind<T> {
    /// Whether a level over `space` must be walked statically: fragments cannot be
    /// indexed by a runtime region (a partition at its partition level). Comptime.
    pub(crate) fn static_level(&self, #[comptime] space: Space) -> comptime_type!(bool) {
        match self {
            TileKind::CmmaPartition(_) => comptime!(partition_level(&space).is_some()),
            TileKind::Gmem(_) | TileKind::Smem(_) | TileKind::Cmma(_) | TileKind::TmaGmem(_) => {
                comptime!(false)
            }
        }
    }

    /// Whether a staged walk at this level must be unrolled for correctness: the level
    /// cuts a fragment partition, so each region selects its block of fragments, which
    /// takes compile-time coordinates. `(1, 1)` counts (a k-step walk) cut nothing and
    /// pass the partition through, so they keep the compact runtime loop. Comptime.
    pub(crate) fn cuts_partition(&self, #[comptime] space: Space) -> comptime_type!(bool) {
        match self {
            TileKind::CmmaPartition(_) => {
                comptime!(matches!(partition_level(&space), Some(c) if c != (1, 1)))
            }
            TileKind::Gmem(_) | TileKind::Smem(_) | TileKind::Cmma(_) | TileKind::TmaGmem(_) => {
                comptime!(false)
            }
        }
    }
}

/// The quantization a tile's backing store carries so reads dequantize transparently. Holds the
/// scales `buffer` plus enough to window it in lockstep with the values ([`MemData::at`]): the
/// per-axis scale `strides`, a running flat `window_start`, and the comptime per-axis `block`
/// edges (elements per block). [`ScaleLayout`] turns that into an address, so the scales read as
/// a view over the values' own window. Per-tensor is the degenerate case: `strides` are all `0`,
/// so `window_start` never leaves `0`.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct QuantInfo {
    pub(crate) buffer: Box<[f32]>,
    pub(crate) strides: Coords<u32>,
    pub(crate) window_start: u32,
    #[cube(comptime)]
    pub(crate) block: Vec<usize>,
    #[cube(comptime)]
    pub scheme: QuantScheme,
}

/// Per-axis block edges (elements per block) for a scheme. Per-tensor is one scale for the whole
/// tensor, so its edges are an unused placeholder ([`QuantInfo::native`] pairs them with `0`
/// strides); a block scheme's edges come straight from the scheme.
pub(crate) fn block_edges(scheme: QuantScheme, rank: usize) -> Vec<usize> {
    match scheme.level {
        QuantLevel::Tensor => vec![1; rank],
        QuantLevel::Block(bs) => bs.to_dim_vec(rank).iter().map(|&b| b as usize).collect(),
    }
}

#[cube]
impl QuantInfo {
    /// Build the native (unpacked) quant side-channel from a launched [`QuantArg`]: capture the
    /// whole scales buffer, plus per logical axis the block edge and the scale stride that indexes
    /// one scale per block. Per-tensor is the degenerate single scale — every stride `0`, so the
    /// window never leaves index `0`; a block scheme reads the scales tensor's own strides.
    pub(crate) fn native(q: &QuantArg, #[comptime] rank: usize) -> QuantInfo {
        let block = comptime!(block_edges(q.scheme, rank));
        let mut strides = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            if comptime!(q.scheme.level == QuantLevel::Tensor) {
                strides.push(0u32);
            } else {
                strides.push(q.scales.stride(p) as u32);
            }
        }
        QuantInfo {
            buffer: unsafe { q.scales.as_slice().as_boxed_unchecked() },
            strides,
            window_start: 0u32,
            block: comptime!(block),
            scheme: comptime!(q.scheme),
        }
    }

    /// Re-window the scales onto a tile whose absolute logical origin is `origin`: the block index
    /// on each axis is `origin / block`, dotted with the scale strides, summed into a flat start.
    /// Element units on every axis (blocks are logical); the inner axis's line origin scales back
    /// by `vector_size`. Per-tensor keeps `strides = 0`, so this stays `0`.
    ///
    /// The window's own block index folds out here so [`ScaleLayout`] only adds the offset within
    /// the window; that split holds because a window never straddles a block
    /// ([`validate_blocks`] rejects a tiling where one would).
    pub(crate) fn window(
        &self,
        origin: &Coords<u32>,
        #[comptime] rank: usize,
        #[comptime] vector_size: usize,
    ) -> QuantInfo {
        let last = comptime!(rank - 1);
        let mut advances = Coords::<u32>::new();
        #[unroll]
        for p in 0..rank {
            let w = comptime!(if p == last { vector_size } else { 1usize });
            let origin_elem = origin.at(p).fmul(comptime!(w as u32).runtime());
            let block = comptime!(self.block[p] as u32).runtime();
            advances.push(origin_elem.fdiv(block).fmul(self.strides.at(p)));
        }
        QuantInfo {
            buffer: unsafe { self.buffer.as_boxed_unchecked() },
            strides: self.strides.clone(),
            window_start: advances.fsum(comptime!((0..rank).collect::<Vec<_>>())),
            block: comptime!(self.block.clone()),
            scheme: comptime!(self.scheme),
        }
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
    /// How this operand's bytes move: a strided cooperative copy or a TMA hardware bulk
    /// copy. Comptime (the kind is fixed at trace); drives the staging sync. A resident
    /// fragment is never a fill source, so it reports strided.
    pub fn delivery(&self) -> comptime_type!(Delivery) {
        match &self.tile_kind {
            TileKind::Gmem(_) | TileKind::Smem(_) => comptime!(Delivery::Strided),
            TileKind::TmaGmem(_) => comptime!(Delivery::Tma),
            TileKind::Cmma(_) | TileKind::CmmaPartition(_) => comptime!(Delivery::Strided),
        }
    }

    /// The [`StagePlan`] a stage derived from this tile takes: its [`StageStorage`] layout
    /// and the launch's cube size. A TMA bulk copy writes its box rows raw and a fragment
    /// is never a fill source, so both report the plain default (strided, units unknown).
    pub fn stage(&self) -> comptime_type!(StagePlan) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.stage(),
            TileKind::TmaGmem(_) | TileKind::Cmma(_) | TileKind::CmmaPartition(_) => {
                comptime!(StagePlan::strided())
            }
        }
    }

    /// Physical vectorization of the backing store: the `Vector<T, vector_size>` line
    /// width the leaf reconstructs. A launched memory tile carries its operand's vector
    /// size; a cmma fragment and a tma source are scalar (`1`).
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
            // A resident fragment passes through unchanged (nothing to window) — legal
            // only on a level that cuts nothing on m/n, like a k-step walk; a cutting
            // level would alias every region onto the one fragment.
            TileKind::Cmma(c) => {
                comptime!(assert!(
                    matches!(partition_level(&self.space), None | Some((1, 1))),
                    "Tile::at: a level that cuts tiles cannot select into a single cmma \
                     fragment (it needs a fragment partition, or a memory output)"
                ));
                TileKind::new_Cmma(c.clone())
            }
            // A partition *selects* under a region with comptime coordinates (an
            // unrolled walk's fold to constants): each region owns a `sub_m × sub_n`
            // block of the fragments — a level that doesn't cut the partition selects
            // the whole of it, a 1×1 block is the fragment itself. A runtime region
            // passes the partition through whole, legal exactly when this level cuts
            // nothing (a k-step walk); the static fragment walk below then selects.
            TileKind::CmmaPartition(p) => {
                let rank = comptime!(self.space.rank());
                let a0 = comptime!(self.space.axis_at(rank - 2));
                let a1 = comptime!(self.space.axis_at(rank - 1));
                // A single-tile static axis (a k-step walk that doesn't cut m/n) folds to a
                // constant `0`, so selection is uniform: a cut axis takes its constant digit,
                // an uncut one selects the whole partition (its one fragment when 1×1). A
                // `Dynamic` axis (only the top level) stays runtime → `None` → walks.
                let mi = if comptime!(self.space.single_static_tile(a0)) {
                    comptime!(Some(0u64))
                } else {
                    region.coord(a0).constant()
                };
                let ni = if comptime!(self.space.single_static_tile(a1)) {
                    comptime!(Some(0u64))
                } else {
                    region.coord(a1).constant()
                };
                match comptime!(mi.zip(ni)) {
                    Some((c0, c1)) => {
                        let (sub_m, sub_n) = comptime!({
                            let (cm, cn) = (self.space.count(a0), self.space.count(a1));
                            assert!(
                                p.m_tiles.is_multiple_of(cm) && p.n_tiles.is_multiple_of(cn),
                                "Tile::at: the level's grid must divide the fragment partition"
                            );
                            (p.m_tiles / cm, p.n_tiles / cn)
                        });
                        let mi = comptime!(c0 as usize * sub_m);
                        let ni = comptime!(c1 as usize * sub_n);
                        if comptime!(sub_m == 1 && sub_n == 1) {
                            TileKind::new_Cmma(p.at(mi, ni))
                        } else {
                            TileKind::new_CmmaPartition(p.window(mi, ni, sub_m, sub_n))
                        }
                    }
                    // A runtime coordinate reaches here only from a `Dynamic` (top, instance)
                    // level, which cuts nothing on m/n and passes the whole partition down to
                    // the static levels below. A rolled *cut* would be a caller bug.
                    None => {
                        comptime!(assert!(
                            partition_level(&self.space).is_none(),
                            "Tile::at: a level that cuts a fragment partition must be \
                             walked with compile-time coordinates (an unrolled walk)"
                        ));
                        TileKind::new_CmmaPartition(p.clone())
                    }
                }
            }
        };
        Tile::<T> {
            tile_kind,
            space: comptime!(self.space.divide()),
        }
    }

    /// This operand's runtime logical size along `axis`, read off the [`bound`](MemData)
    /// folded from the tensor shape. The source of a [`Dynamic`](crate::Extent) axis's
    /// tile count. A cmma fragment has no buffer extent.
    pub fn runtime_extent(&self, #[comptime] axis: Axis) -> usize {
        let p = comptime!(self.space.position(axis));
        let raw = match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => g.bound.at(p).fcast::<usize>(),
            TileKind::TmaGmem(t) => t.bound[p].fcast::<usize>(),
            TileKind::Cmma(_) | TileKind::CmmaPartition(_) => {
                panic!("Tile::runtime_extent: a cmma fragment has no extent")
            }
        };
        // `bound` is a line count on the vectorized innermost axis; the walk divides by
        // conceptual edges, so return line count × width.
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

    /// Blocking copy of `src` into `self`, each kind pairing dispatched to its kind's
    /// transport leaf. A partition source is matched first: it needs the whole
    /// destination tile, which the pairing match below would keep borrowed.
    pub fn copy_from(&mut self, src: &Tile<T>) {
        match &src.tile_kind {
            TileKind::CmmaPartition(s) => s.drain_into(self),
            TileKind::Gmem(_) | TileKind::Smem(_) | TileKind::Cmma(_) | TileKind::TmaGmem(_) => {
                match (&mut self.tile_kind, &src.tile_kind) {
                    (TileKind::CmmaPartition(d), TileKind::Gmem(_) | TileKind::Smem(_)) => {
                        d.fill_from(src)
                    }
                    (TileKind::Cmma(d), TileKind::Gmem(s) | TileKind::Smem(s)) => d.load_window(s),
                    (TileKind::Gmem(d) | TileKind::Smem(d), TileKind::Cmma(s)) => s.store_window(d),
                    (TileKind::Smem(d), TileKind::TmaGmem(s)) => s.load_into(d),
                    (
                        TileKind::Gmem(d) | TileKind::Smem(d),
                        TileKind::Gmem(s) | TileKind::Smem(s),
                    ) => d.fill_from(s),
                    (TileKind::Cmma(_), TileKind::Cmma(_)) => {
                        panic!("Tile::copy_from: cmma→cmma cast not wired")
                    }
                    _ => panic!("Tile::copy_from: unsupported kind pairing"),
                }
            }
        }
    }

    /// Drain a resident accumulator into memory `dst`, casting `T` down to `dst`'s
    /// element type. The cross-type epilogue [`copy_from`](Self::copy_from) cannot express:
    /// its memory transports move bytes and so stay same-type, but a register accumulator
    /// (e.g. `f32`) is wider than the output it writes (e.g. `f16`). Only a fragment
    /// partition drains this way.
    pub fn drain_cast_into<Out: Numeric>(&self, dst: &mut Tile<Out>) {
        match &self.tile_kind {
            TileKind::CmmaPartition(s) => s.drain_cast_into(dst),
            TileKind::Gmem(_) | TileKind::Smem(_) | TileKind::Cmma(_) | TileKind::TmaGmem(_) => {
                panic!("Tile::drain_cast_into: only a fragment partition drains with a cast")
            }
        }
    }
}
