//! The [`Tile`]: one operand's data as a [`TileKind`] backing store, plus the comptime
//! [`Space`] it projects. Structure only; each store's own data and leaves live in its file
//! ([`mem`], [`cmma`], [`tma`]). The launch surface (args, deliveries, builder) rides on
//! [`StridedTileArg`](crate::StridedTileArg) and its twin [`TmaTileArg`](crate::TmaTileArg).

mod cmma;
mod mem;
mod mma;
mod plane;
mod tma;
mod view;

pub use cmma::*;
pub use mem::*;
pub use mma::*;
pub use plane::*;
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
    /// One plane-level tile: owned by a plane and sliced across its lanes, so never addressable
    /// (no memory view). The [`Leaf`] picks the encoding; the contraction is its own.
    PlaneTile(PlaneTile<T>),
    /// The grid of plane tiles one plane owns, `m_tiles × n_tiles`, comptime-indexed; only a
    /// static walk's regions (constant coordinates) can select through it.
    PlanePartition(PlanePartition<T>),
    /// A TMA tensor-map source: not element-addressable, its only sink is a hardware bulk
    /// copy into shared memory. Launched via [`TmaTileArg`], the twin of [`StridedTileArg`].
    TmaGmem(TmaData<T>),
}

/// Where a tile's store lives, so [`zero`](Tile::zero) knows how to clear it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Residence {
    /// Register-resident: a plane tile or a partition of them, cleared in place.
    Fragment,
    /// Memory-backed: gmem, smem, or a tma source, cleared through the walk.
    Memory,
}

#[cube]
impl<T: Numeric> TileKind<T> {
    /// Whether a level must be walked with comptime coordinates. Fragments can't be picked out by
    /// a runtime region, so a plane partition sitting at a real partition level forces the unrolled
    /// walk. Comptime.
    pub(crate) fn static_level(&self, #[comptime] space: Space) -> comptime_type!(bool) {
        match self {
            TileKind::PlanePartition(_) => {
                comptime!(matches!(plane_level(&space), PlaneLevel::Partition { .. }))
            }
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_) => {
                comptime!(false)
            }
        }
    }

    /// Whether a staged walk here must unroll for correctness. When the level cuts a fragment
    /// partition, each region picks its own block of fragments, which needs comptime coordinates.
    /// A 1×1 level (a k-step walk) cuts nothing and passes the partition through, so it stays a
    /// plain runtime loop. Comptime.
    pub(crate) fn cuts_partition(&self, #[comptime] space: Space) -> comptime_type!(bool) {
        match self {
            TileKind::PlanePartition(_) => {
                comptime!(
                    matches!(plane_level(&space), PlaneLevel::Partition { m, n } if (m, n) != (1, 1))
                )
            }
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_) => {
                comptime!(false)
            }
        }
    }

    /// Where this store lives: register [`Fragment`](Residence::Fragment) or
    /// [`Memory`](Residence::Memory). Comptime.
    pub(crate) fn residence(&self) -> comptime_type!(Residence) {
        match self {
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => comptime!(Residence::Fragment),
            TileKind::Gmem(_) | TileKind::Smem(_) | TileKind::TmaGmem(_) => {
                comptime!(Residence::Memory)
            }
        }
    }
}

/// Quantization a tile's store carries, so reads dequantize on their own.
///
/// It holds the scale `buffer` plus what walks the scales in step with the values: a per-axis
/// `strides`, a running `window_start`, and the comptime `block` sizes (elements per block).
/// [`ScaleLayout`] turns those into an address, so a scale reads as a view over the value's own
/// window ([`MemData::at`]).
///
/// Per-tensor quant is the trivial case: one scale for everything, so every stride is `0` and
/// `window_start` never moves.
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
    /// Build the native (unpacked) quant side-channel from a launched [`QuantArg`].
    ///
    /// Captures the whole scales buffer, and per axis the block size plus the stride that steps one
    /// scale per block. A block scheme reads those strides off the scales tensor; per-tensor has a
    /// single scale, so every stride is `0` and the window stays put.
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

    /// Re-window the scales onto a tile whose absolute logical origin is `origin`.
    ///
    /// Per axis the block index is `origin / block`; dot that with the scale strides and sum for a
    /// flat start. Units are elements everywhere (blocks are logical), and the inner axis's line
    /// origin scales back by `vector_size`. Per-tensor keeps every stride at `0`, so the start stays `0`.
    ///
    /// Folding the window's own block index in here lets [`ScaleLayout`] add only the offset within
    /// the window. That split is sound because a window never straddles a block, which
    /// `validate_scheme` enforces.
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

/// One operand's data: a runtime [`TileKind`] backing store and the comptime [`Space`] it projects.
///
/// `T` is the element the tile serves and computes in. Its physical vector width is a storage
/// detail, kept inside the [`TileKind`] and read back with [`vector_size`](Tile::vector_size).
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
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => comptime!(Delivery::Strided),
        }
    }

    /// The [`StagePlan`] a stage derived from this tile takes: its [`StageStorage`] layout
    /// and the launch's cube size. A TMA bulk copy writes its box rows raw and a fragment
    /// is never a fill source, so both report the plain default (strided, units unknown).
    pub fn stage(&self) -> comptime_type!(StagePlan) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.stage(),
            TileKind::TmaGmem(_) | TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
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
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) | TileKind::TmaGmem(_) => {
                comptime!(1usize)
            }
        }
    }

    /// Comptime quant dispatch for a leaf read (`0` = plain, `1` = native i8, `>1` = packed u32);
    /// see [`MemData::quant_pack`]. A resident fragment and a tma source are never quantized.
    pub(crate) fn quant_pack(&self) -> comptime_type!(usize) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.quant_pack(),
            TileKind::TmaGmem(_) | TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                comptime!(0usize)
            }
        }
    }

    /// Window this tile down to `region`, no copy.
    ///
    /// Each tile projects `region` onto its own axes, so `lhs ∈ {M,K}` and `out ∈ {M,N}` line up
    /// on their own; the caller never matches axes by hand.
    pub fn at(&self, region: &Region) -> Tile<T> {
        let tile_kind = match &self.tile_kind {
            TileKind::Gmem(g) => TileKind::new_Gmem(g.at(region, comptime!(self.space.clone()))),
            TileKind::Smem(g) => TileKind::new_Smem(g.at(region, comptime!(self.space.clone()))),
            TileKind::TmaGmem(t) => {
                TileKind::new_TmaGmem(t.at(region, comptime!(self.space.clone())))
            }
            // A plane tile has nothing to window: pass it through. Legal only where the level
            // cuts nothing on m/n (a k-step walk); a cutting level would alias every region
            // onto the one tile.
            TileKind::PlaneTile(t) => {
                comptime!(assert!(
                    matches!(
                        plane_level(&self.space),
                        PlaneLevel::Final
                            | PlaneLevel::Instance
                            | PlaneLevel::Partition { m: 1, n: 1 }
                    ),
                    "Tile::at: a level that cuts tiles cannot select into a single plane \
                     tile (it needs a partition, or a memory output)"
                ));
                TileKind::new_PlaneTile(t.clone())
            }
            // A partition selects under comptime coordinates (an unrolled walk folds regions
            // to constants): each region owns a `sub_m × sub_n` block. An uncut level selects
            // the whole partition; a 1×1 block is the tile itself. A runtime region passes the
            // partition through whole, legal only on an uncut k-step level (the walk below
            // then selects statically).
            TileKind::PlanePartition(p) => {
                let rank = comptime!(self.space.rank());
                let a0 = comptime!(self.space.axis_at(rank - 2));
                let a1 = comptime!(self.space.axis_at(rank - 1));
                // A single-tile static axis (k-step, no m/n cut) folds to constant `0`, so a
                // cut axis takes its constant digit and an uncut one selects the whole
                // partition. A `Dynamic` axis (top level only) stays runtime, yielding `None`.
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
                                "Tile::at: the level's grid must divide the partition"
                            );
                            (p.m_tiles / cm, p.n_tiles / cn)
                        });
                        let mi = comptime!(c0 as usize * sub_m);
                        let ni = comptime!(c1 as usize * sub_n);
                        if comptime!(sub_m == 1 && sub_n == 1) {
                            TileKind::new_PlaneTile(p.at(mi, ni))
                        } else {
                            TileKind::new_PlanePartition(p.window(mi, ni, sub_m, sub_n))
                        }
                    }
                    // A runtime coordinate reaches here only from a `Dynamic` (top, instance)
                    // level, which cuts nothing on m/n and passes the whole partition down to
                    // the static levels below. A rolled *cut* would be a caller bug.
                    None => {
                        comptime!(assert!(
                            matches!(
                                plane_level(&self.space),
                                PlaneLevel::Final | PlaneLevel::Instance
                            ),
                            "Tile::at: a level that cuts a partition must be \
                             walked with compile-time coordinates (an unrolled walk)"
                        ));
                        TileKind::new_PlanePartition(p.clone())
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
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Tile::runtime_extent: a plane tile has no extent")
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

    /// Zero this tile in place: `mma` accumulates over whatever is there, so a routine
    /// whose contract is `out = A·B` zeroes first. Memory recurses the walk (each
    /// instance zeroes exactly the windows its `mma` owns); fragments fill in place.
    pub fn zero(&mut self) {
        let residence = self.tile_kind.residence();
        match comptime!(residence) {
            Residence::Fragment => match &mut self.tile_kind {
                TileKind::PlaneTile(t) => t.zero(),
                TileKind::PlanePartition(p) => p.zero(),
                TileKind::Gmem(_) | TileKind::Smem(_) | TileKind::TmaGmem(_) => {
                    panic!("Tile::zero: a fragment residence holds no memory store")
                }
            },
            Residence::Memory => match comptime!(self.space.partitioner().clone()) {
                // A final memory tile is one window: clear it directly.
                Partitioner::Final(_) => match &mut self.tile_kind {
                    TileKind::Gmem(d) | TileKind::Smem(d) => d.zero(),
                    TileKind::TmaGmem(_) => panic!("Tile::zero: a tma source is not writable"),
                    TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                        panic!("Tile::zero: a memory residence holds no fragment")
                    }
                },
                // A level recurses so each region clears exactly the windows its `mma` owns.
                Partitioner::Level(_) => {
                    for region in Walk::over(self.runtime_space()) {
                        let mut sub = self.at(&region);
                        sub.zero();
                    }
                }
            },
        }
    }

    /// Blocking copy of `src` into `self`, each kind pairing dispatched to its kind's
    /// transport leaf. A partition source is matched first: it needs the whole
    /// destination tile, which the pairing match below would keep borrowed.
    pub fn copy_from(&mut self, src: &Tile<T>) {
        match &src.tile_kind {
            TileKind::PlanePartition(s) => s.drain_into(self),
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_) => match (&mut self.tile_kind, &src.tile_kind) {
                (TileKind::PlanePartition(d), TileKind::Gmem(_) | TileKind::Smem(_)) => {
                    d.fill_from(src)
                }
                (TileKind::PlaneTile(d), TileKind::Gmem(s) | TileKind::Smem(s)) => d.load_window(s),
                (TileKind::Gmem(d) | TileKind::Smem(d), TileKind::PlaneTile(s)) => {
                    s.store_window(d)
                }
                (TileKind::Smem(d), TileKind::TmaGmem(s)) => s.load_into(d),
                (TileKind::Gmem(d) | TileKind::Smem(d), TileKind::Gmem(s) | TileKind::Smem(s)) => {
                    d.fill_from(s)
                }
                (TileKind::PlaneTile(_), TileKind::PlaneTile(_)) => {
                    panic!("Tile::copy_from: plane tile to plane tile cast not wired")
                }
                _ => panic!("Tile::copy_from: unsupported kind pairing"),
            },
        }
    }

    /// Drain a resident accumulator into memory `dst`, casting `T` down to `dst`'s element
    /// type. [`copy_from`](Self::copy_from) can't: its transports move bytes so stay same-type,
    /// but a register accumulator (`f32`) is wider than the output it writes (`f16`). Only a
    /// fragment partition drains this way.
    pub fn drain_cast_into<Out: Numeric>(&self, dst: &mut Tile<Out>) {
        match &self.tile_kind {
            TileKind::PlanePartition(s) => s.drain_cast_into(dst),
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::TmaGmem(_) => {
                panic!("Tile::drain_cast_into: only a partition drains with a cast")
            }
        }
    }
}
