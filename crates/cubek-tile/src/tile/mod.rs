//! The [`Tile`]: one operand's data, a [`TileKind`] backing store plus the comptime
//! [`Space`] it projects. Kernel-side structure only; each backing store's own data and
//! leaves live in its file ([`mem`], [`cmma`], [`tma`]), and the launch surface (the
//! arguments, deliveries and builder a tile is loaded through) lives in [`crate::load`].

mod cmma;
mod mem;
mod tma;

pub use cmma::*;
pub use mem::*;
pub use tma::*;

use cubecl::{prelude::*, quant::scheme::QuantScheme};

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
    /// A partition of cmma fragments, `m_tiles × n_tiles`, comptime-indexed; walked
    /// statically ([`at_static`](Tile::at_static)).
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
}

/// The quantization a tile's backing store carries so reads dequantize transparently: a
/// runtime `scale` (per-tensor for now) plus the comptime [`QuantScheme`].
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct QuantInfo {
    pub scale: f32,
    #[cube(comptime)]
    pub scheme: QuantScheme,
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

    /// The [`StageStorage`] layout a stage derived from this tile takes. A TMA bulk copy
    /// writes its box rows raw, so its stages stay plain strided.
    pub fn stage_storage(&self) -> comptime_type!(StageStorage) {
        match &self.tile_kind {
            TileKind::Gmem(d) | TileKind::Smem(d) => d.stage,
            TileKind::TmaGmem(_) | TileKind::Cmma(_) | TileKind::CmmaPartition(_) => {
                comptime!(StageStorage::Strided)
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
            // A resident fragment (or partition) passes through unchanged: a runtime
            // region cannot select fragments. At the partition level, `at_static` selects.
            TileKind::Cmma(c) => TileKind::new_Cmma(c.clone()),
            TileKind::CmmaPartition(p) => TileKind::new_CmmaPartition(p.clone()),
        };
        Tile::<T> {
            tile_kind,
            space: comptime!(self.space.divide()),
        }
    }

    /// [`at`](Tile::at) for a static region: the register tier's windowing. Memory
    /// windows identically (the coordinates coerce to a runtime [`Region`]); a fragment
    /// partition *selects* its `(mi, ni)` fragment, which only static coordinates can do.
    pub fn at_static(&self, #[comptime] region: &StaticRegion) -> Tile<T> {
        match &self.tile_kind {
            TileKind::Gmem(_) | TileKind::Smem(_) | TileKind::TmaGmem(_) => {
                self.at(&Region::from_static(region))
            }
            TileKind::CmmaPartition(p) => {
                let mi = comptime!(region.coord(self.space.axis_at(self.space.rank() - 2)));
                let ni = comptime!(region.coord(self.space.axis_at(self.space.rank() - 1)));
                Tile::<T> {
                    tile_kind: TileKind::new_Cmma(p.at(mi, ni)),
                    space: comptime!(self.space.divide()),
                }
            }
            TileKind::Cmma(_) => panic!("Tile::at_static: a single fragment has no regions"),
        }
    }

    /// This operand's runtime logical size along `axis`, read off the [`bound`](MemData)
    /// folded from the tensor shape. The source of a [`Dynamic`](crate::Extent) axis's
    /// tile count. A cmma fragment has no buffer extent.
    pub fn runtime_extent(&self, #[comptime] axis: Axis) -> usize {
        let p = comptime!(self.space.position(axis));
        let raw = match &self.tile_kind {
            TileKind::Gmem(g) | TileKind::Smem(g) => g.bound[p] as usize,
            TileKind::TmaGmem(t) => t.bound[p] as usize,
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
}
