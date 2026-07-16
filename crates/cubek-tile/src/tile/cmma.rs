//! The tensor-core backing stores: a single fragment ([`CmmaData`]) and a partition of
//! them ([`CmmaPartition`]), plus the fragment↔memory transports.

use cubecl::{
    cmma::{self, Matrix, MatrixIdent, MatrixLayout},
    prelude::*,
};

use crate::*;

/// A tensor-core fragment plus the comptime config its load/store paths dispatch on.
/// `Clone` duplicates the handle, not the fragment: a clone is the same matrix.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct CmmaData<T: Numeric> {
    pub matrix: Matrix<T>,
    #[cube(comptime)]
    pub ident: MatrixIdent,
    #[cube(comptime)]
    pub layout: MatrixLayout,
}

/// A partition of cmma fragments: `m_tiles × n_tiles` over the tile's trailing two axes,
/// row-major comptime-indexed (`mi · n_tiles + ni`). Mirrors cubek-std's `PartitionTile`.
/// `Clone` duplicates the handles, not the fragments.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct CmmaPartition<T: Numeric> {
    pub frags: Sequence<CmmaData<T>>,
    #[cube(comptime)]
    pub m_tiles: usize,
    #[cube(comptime)]
    pub n_tiles: usize,
}

#[cube]
impl<T: Numeric> CmmaPartition<T> {
    /// The `(mi, ni)` fragment (a handle clone). Comptime indices only: fragments
    /// cannot be selected at runtime.
    pub(crate) fn at(&self, #[comptime] mi: usize, #[comptime] ni: usize) -> CmmaData<T> {
        self.frags.index(comptime!(mi * self.n_tiles + ni)).clone()
    }

    /// The `m_tiles × n_tiles` sub-partition at `(mi, ni)` (handle clones, so its
    /// fragments are the parent's): a stacked partition level selects a block of
    /// fragments where the fragment grid itself selects one.
    pub(crate) fn window(
        &self,
        #[comptime] mi: usize,
        #[comptime] ni: usize,
        #[comptime] m_tiles: usize,
        #[comptime] n_tiles: usize,
    ) -> CmmaPartition<T> {
        let mut frags = Sequence::<CmmaData<T>>::new();
        #[unroll]
        for i in 0..m_tiles {
            #[unroll]
            for j in 0..n_tiles {
                frags.push(self.at(comptime!(mi + i), comptime!(ni + j)));
            }
        }
        CmmaPartition::<T> {
            frags,
            m_tiles,
            n_tiles,
        }
    }

    /// The register form of an accumulator over `space`: a partition mirroring its
    /// fragment grid, each fragment zeroed (`c = a·b` starts from the additive
    /// identity, `beta = 0`). `k` is the instruction's contraction depth (the space's
    /// own axes only give `m`/`n`).
    pub(crate) fn mirror(#[comptime] space: Space, #[comptime] k: usize) -> Tile<T> {
        let (m_tiles, n_tiles) = comptime!(partition_shape(&space));
        let fin = comptime!(space.final_space());
        let m = comptime!(fin.extent_at(fin.rank() - 2));
        let n = comptime!(fin.extent_at(fin.rank() - 1));

        let mut frags = Sequence::<CmmaData<T>>::new();
        #[unroll]
        for _mi in 0..m_tiles {
            #[unroll]
            for _ni in 0..n_tiles {
                let mut frag = CmmaData::<T>::alloc(MatrixIdent::Accumulator, m, n, k);
                cmma::fill(&mut frag.matrix, T::from_int(0));
                frags.push(frag);
            }
        }
        Tile::<T> {
            tile_kind: TileKind::new_CmmaPartition(CmmaPartition::<T> {
                frags,
                m_tiles,
                n_tiles,
            }),
            space: comptime!(space),
        }
    }

    /// The staging store for one region of an operand under `out`'s contraction: a
    /// partition mirroring the region's fragment grid, fragments uninitialized;
    /// [`copy_from`](Tile::copy_from) fills it.
    pub(crate) fn store(#[comptime] window: Space, #[comptime] out: Space) -> Tile<T> {
        let a0 = comptime!(window.axis_at(window.rank() - 2));
        let a1 = comptime!(window.axis_at(window.rank() - 1));
        let t0 = comptime!(window.count(a0));
        let t1 = comptime!(window.count(a1));

        // `A` is `m×k`, `B` is `k×n`: the operand's role is where its contracted axis sits.
        let contracted = comptime!(window.contraction(&out));
        let ident = comptime!(if contracted == a1 {
            MatrixIdent::A
        } else {
            assert!(
                contracted == a0,
                "CmmaPartition::store: the contracted axis must be one of the trailing two"
            );
            MatrixIdent::B
        });
        let out_fin = comptime!(out.final_space());
        let m = comptime!(out_fin.extent_at(out_fin.rank() - 2));
        let n = comptime!(out_fin.extent_at(out_fin.rank() - 1));
        let k = comptime!(window.final_space().extent(contracted));

        let mut frags = Sequence::<CmmaData<T>>::new();
        #[unroll]
        for _i in 0..t0 {
            #[unroll]
            for _j in 0..t1 {
                frags.push(CmmaData::<T>::alloc(ident, m, n, k));
            }
        }
        Tile::<T> {
            tile_kind: TileKind::new_CmmaPartition(CmmaPartition::<T> {
                frags,
                m_tiles: t0,
                n_tiles: t1,
            }),
            space: comptime!(window),
        }
    }

    /// Fill each fragment from its final window of `src`, in the partition's row-major
    /// fragment order.
    pub(crate) fn fill_from(&self, src: &Tile<T>) {
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let mut frag = self.at(mi, ni);
                let window = src.fragment_window(mi, ni);
                match &window.tile_kind {
                    TileKind::Gmem(g) | TileKind::Smem(g) => frag.load_window(g),
                    TileKind::Cmma(_) | TileKind::CmmaPartition(_) | TileKind::TmaGmem(_) => {
                        panic!("CmmaPartition::fill_from: the source must be memory")
                    }
                }
            }
        }
    }

    /// Drain each fragment into its final window of `dst`; [`fill_from`](Self::fill_from)'s
    /// inverse.
    pub(crate) fn drain_into(&self, dst: &mut Tile<T>) {
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let frag = self.at(mi, ni);
                let mut window = dst.fragment_window(mi, ni);
                match &mut window.tile_kind {
                    TileKind::Gmem(g) | TileKind::Smem(g) => frag.store_window(g),
                    TileKind::Cmma(_) | TileKind::CmmaPartition(_) | TileKind::TmaGmem(_) => {
                        panic!("CmmaPartition::drain_into: the sink must be memory")
                    }
                }
            }
        }
    }

    /// Drain each fragment into its final window of `dst`, casting `T` to `dst`'s element
    /// type first. The cross-type epilogue: a register accumulator (e.g. `f32`) written to
    /// a narrower output (e.g. `f16`). Same as [`drain_into`](Self::drain_into) when the
    /// types match (the cast is a no-op).
    pub(crate) fn drain_cast_into<Out: Numeric>(&self, dst: &mut Tile<Out>) {
        #[unroll]
        for mi in 0..comptime!(self.m_tiles) {
            #[unroll]
            for ni in 0..comptime!(self.n_tiles) {
                let frag = self.at(mi, ni);
                let mut window = dst.fragment_window(mi, ni);
                match &mut window.tile_kind {
                    TileKind::Gmem(g) | TileKind::Smem(g) => frag.store_cast_window(g),
                    TileKind::Cmma(_) | TileKind::CmmaPartition(_) | TileKind::TmaGmem(_) => {
                        panic!("CmmaPartition::drain_cast_into: the sink must be memory")
                    }
                }
            }
        }
    }
}

#[cube]
impl<T: Numeric> Tile<T> {
    /// Descend to the `(mi, ni)` fragment's final window: an instance level hands this
    /// instance a single region; a partition level takes its own digit of the fragment
    /// coordinates — the grid may be split across stacked levels, so each consumes the
    /// high digits (the levels below it are the place value) and passes the rest down.
    fn fragment_window(&self, #[comptime] mi: usize, #[comptime] ni: usize) -> Tile<T> {
        let space = comptime!(self.space.clone());
        match comptime!(partition_level(&space)) {
            None => {
                let walk = Walk::over(self.runtime_space());
                let sub = self.at(&walk.region(0));
                match comptime!(sub.space.partitioner()) {
                    Partitioner::Final(_) => sub,
                    Partitioner::Level(_) => sub.fragment_window(mi, ni),
                }
            }
            Some(_) => {
                let (bm, bn) = comptime!(partition_shape(&space.divide()));
                let region = Region::trailing(
                    comptime!(space.clone()),
                    comptime!(mi / bm),
                    comptime!(ni / bn),
                );
                let sub = self.at(&region);
                match comptime!(sub.space.partitioner()) {
                    Partitioner::Final(_) => sub,
                    Partitioner::Level(_) => {
                        sub.fragment_window(comptime!(mi % bm), comptime!(ni % bn))
                    }
                }
            }
        }
    }
}

/// The per-instance tile count of `axis` at this level, `None` when it is runtime.
fn per_instance_tiles(level: &Space, axis: Axis) -> Option<usize> {
    let edge = level.partitioner().edge(axis);
    match level.partitioner().distribution(axis) {
        Distribution::Sequential => match level.extent_raw(axis) {
            Extent::Static(e) => Some(e.div_ceil(edge)),
            Extent::Dynamic => None,
        },
        Distribution::Spatial { coverage, .. } => match coverage {
            Coverage::TilesEach(t) => Some(t),
            Coverage::Instances(n) => match level.extent_raw(axis) {
                Extent::Static(e) => Some(e.div_ceil(edge).div_ceil(n)),
                Extent::Dynamic => None,
            },
        },
    }
}

/// Classify the current level of a space that backs fragments: `None` for an *instance*
/// level (a `Spatial` split across hardware, one tile per instance), or the trailing-two-axes
/// tile counts for the *partition* level — a purely sequential level is one even at a 1×1
/// grid (cuts equal to the level below still back per-instance fragments). Anything else
/// cannot back fragments and panics at comptime.
pub(crate) fn partition_level(space: &Space) -> Option<(usize, usize)> {
    if space.is_final() {
        return None;
    }
    if space.axes().any(|a| {
        matches!(
            space.partitioner().distribution(a),
            Distribution::Spatial { .. }
        )
    }) {
        for axis in space.axes() {
            assert!(
                per_instance_tiles(space, axis) == Some(1),
                "cmma instance level: every axis must hand out one tile"
            );
        }
        return None;
    }
    let rank = space.rank();
    for (p, axis) in space.axes().enumerate() {
        let tiles = per_instance_tiles(space, axis)
            .expect("cmma partition level: tile counts must be comptime");
        assert!(
            p >= rank - 2 || tiles == 1,
            "cmma partition level: leading (batch) axes must hand out one tile"
        );
    }
    Some((
        per_instance_tiles(space, space.axis_at(rank - 2)).unwrap(),
        per_instance_tiles(space, space.axis_at(rank - 1)).unwrap(),
    ))
}

/// The whole remaining walk's fragment grid for one instance: `(1, 1)` when every level
/// is an instance level, else the componentwise product of the partition levels' tile
/// counts (the grid may be split across stacked levels, e.g. an N-walk staging level
/// over an M-only static walk).
fn partition_shape(space: &Space) -> (usize, usize) {
    let mut shape = (1usize, 1usize);
    let mut level = space.clone();
    while !level.is_final() {
        if let Some((m, n)) = partition_level(&level) {
            shape = (shape.0 * m, shape.1 * n);
        }
        level = level.divide();
    }
    shape
}

#[cube]
impl<T: Numeric> CmmaData<T> {
    /// Allocate an uninitialized fragment. `m`/`n`/`k` are the whole MMA tile, passed in
    /// full whatever the role; the layout is `RowMajor` (how the stages are laid out).
    pub(crate) fn alloc(
        #[comptime] ident: MatrixIdent,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] k: usize,
    ) -> CmmaData<T> {
        let matrix = unsafe { Matrix::<T>::uninitialized(ident, m, n, k, MatrixLayout::RowMajor) };
        CmmaData::<T> {
            matrix,
            ident,
            layout: MatrixLayout::RowMajor,
        }
    }

    /// An uninitialized fragment presented as a `Cmma` tile. `m`/`n`/`k` are the whole
    /// MMA tile, passed in full whatever the role.
    pub fn fragment(
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

    /// Fill this fragment from `mem`'s *window*: `A`/`B` use `cmma::load`, an
    /// `Accumulator` uses `load_with_layout`. Rows step by the store's physical row
    /// stride, so a window into a larger stage loads like a whole buffer.
    pub(crate) fn load_window(&mut self, mem: &MemData<T>) {
        let stride = mem.row_stride();
        match comptime!(self.ident) {
            MatrixIdent::Accumulator => cmma::load_with_layout(
                &mut self.matrix,
                mem.window_slice(),
                stride,
                comptime!(self.layout),
            ),
            _ => cmma::load(&mut self.matrix, mem.window_slice(), stride),
        }
    }

    /// Drain this fragment into `mem`'s *window* (origin offset, physical row stride).
    pub(crate) fn store_window(&self, mem: &mut MemData<T>) {
        let stride = mem.row_stride();
        cmma::store(
            mem.window_slice_mut(),
            &self.matrix,
            stride,
            comptime!(self.layout),
        )
    }

    /// Drain this fragment into `mem`'s *window*, casting `T` down to the sink's element
    /// type first: a register accumulator (e.g. `f32`) is wider than the stored output
    /// (e.g. `f16`). The cast is a no-op when the types match.
    pub(crate) fn store_cast_window<Out: Numeric>(&self, mem: &mut MemData<Out>) {
        let stride = mem.row_stride();
        let casted: Matrix<Out> = cmma::cast(&self.matrix);
        cmma::store(
            mem.window_slice_mut(),
            &casted,
            stride,
            comptime!(self.layout),
        )
    }
}
