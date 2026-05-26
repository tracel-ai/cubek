use cubecl;
use cubecl::prelude::*;

use crate::StageIdent;
use crate::tile::{
    Plane, RowWise, Tile, TileKind, TileKindExpand,
    mask::Mask,
    scope::{TileScope, assert_plane_scope},
    variants::{
        instruction::cmma::CmmaTile,
        whitebox_fragment::{InnerLayout, WhiteboxFragment, WhiteboxFragmentLayout},
    },
};

/// Comptime configuration for [`BounceTile`].
///
/// A bounce tile bundles an opaque cmma fragment together with a shared-memory
/// scratch slice and a [`WhiteboxFragment`] view, so row-wise operations can be
/// expressed as `copy_from` between the inner pieces. From the caller's point
/// of view it is a single [`Tile`] variant — only valid when the tile's
/// scope is `Plane`.
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct BounceConfig {
    pub tile_shape: (u32, u32),
    pub num_planes: u32,
    pub plane_dim: u32,
    pub inner_layout: InnerLayout,
}

#[derive(CubeType)]
pub struct BounceTile<N: Numeric> {
    pub cmma: CmmaTile<N>,
    pub smem: Shared<[N]>,
    pub fragment: WhiteboxFragment<N>,
}

#[cube]
impl<N: Numeric> BounceTile<N> {
    pub fn new(cmma: CmmaTile<N>, #[comptime] cfg: BounceConfig) -> BounceTile<N> {
        let total_tile_size = comptime!((cfg.tile_shape.0 * cfg.tile_shape.1) as usize);
        let smem_size = comptime!(total_tile_size * cfg.num_planes as usize);
        let start = UNIT_POS_Y as usize * total_tile_size;
        let end = start + total_tile_size;
        let smem = Shared::new_slice(smem_size).map(|smem| &smem[start..end]);

        let layout = comptime!(WhiteboxFragmentLayout::new(
            cfg.tile_shape,
            cfg.plane_dim,
            cfg.inner_layout
        ));
        let fragment = WhiteboxFragment::new(layout);

        BounceTile::<N> {
            cmma,
            smem,
            fragment,
        }
    }
}

#[cube]
impl<E: Float> BounceTile<E> {
    /// Synchronizes the fragment view from the cmma fragment via smem.
    /// Call before any rowwise/elementwise op so the fragment reflects the
    /// current cmma state.
    pub fn cmma_to_fragment(&mut self) {
        let stride = comptime!(self.cmma.tile_size.n());
        cubecl::cmma::store(
            &mut self.smem,
            &self.cmma.matrix,
            stride,
            cubecl::cmma::MatrixLayout::RowMajor,
        );
        sync_cube();
        self.fragment.load_from_slice(&self.smem);
        sync_cube();
    }

    /// Synchronizes the cmma fragment from the fragment view via smem. Call
    /// after rowwise/elementwise edits to make the cmma side current for the
    /// next mma.
    pub fn fragment_to_cmma(&mut self) {
        let stride = comptime!(self.cmma.tile_size.n());
        self.fragment.store_to(&mut self.smem);
        sync_cube();
        cubecl::cmma::load_with_layout(
            &mut self.cmma.matrix,
            &self.smem,
            stride,
            cubecl::cmma::MatrixLayout::RowMajor,
        );
    }

    pub fn row_max(&self, acc: &mut RowWise<E>, base: &RowWise<E>) {
        self.fragment.row_max(acc, base);
    }

    pub fn row_sum(&self, acc: &mut RowWise<E>) {
        self.fragment.row_sum(acc);
    }

    pub fn exp_diff(&mut self, rowwise: &RowWise<E>) {
        self.fragment.exp_diff(rowwise);
    }

    pub fn rowwise_scale(&mut self, scale: &RowWise<E>) {
        self.fragment.rowwise_scale(scale);
    }

    pub fn scale_and_mask<M: Mask>(&mut self, scale: E, mask: &M) {
        self.fragment.scale_and_mask::<M>(scale, mask);
    }

    /// Zeros the cmma fragment. The fragment view is not the live storage at
    /// fill_zero call sites (always invoked before any cmma_to_fragment), so
    /// only cmma needs clearing.
    pub fn fill_zero(&mut self) {
        cubecl::cmma::fill(&mut self.cmma.matrix, E::from_int(0));
    }

    /// Writes the (already-softmaxed) fragment view of this bounce tile into
    /// `softmaxed`. The source fragment is plane-fragmented; for a `Bounce`
    /// destination this routes through the destination's smem into its cmma
    /// fragment.
    pub fn write_fragment_to<Lhs: Float, Sc: TileScope>(&self, softmaxed: &mut Tile<Lhs, Sc>) {
        write_fragment_into::<E, Lhs, Sc>(&self.fragment, softmaxed);
    }
}

#[cube]
fn write_fragment_into<Acc: Float, Lhs: Float, Sc: TileScope>(
    src: &WhiteboxFragment<Acc>,
    softmaxed: &mut Tile<Lhs, Sc>,
) {
    match &mut softmaxed.kind {
        TileKind::Bounce(d) => {
            let stride = comptime!(d.cmma.tile_size.n());
            src.store_to(&mut d.smem);
            sync_cube();
            cubecl::cmma::load(&mut d.cmma.matrix, &d.smem, stride);
        }
        TileKind::WhiteboxFragment(d) => {
            let total = comptime!(src.layout.unit_size.0 * src.layout.unit_size.1);
            for i in 0..total {
                d.array[i as usize] = Lhs::cast_from(src.array[i as usize]);
            }
        }
        _ => panic!("write_fragment_to: unsupported softmaxed variant"),
    }
}

#[cube]
/// Wraps a freshly built `CmmaTile` in a `Tile::Bounce`. Panics at expansion
/// time unless `Sc = Plane`.
pub fn allocate_bounce_tile<E: Numeric, Sc: TileScope>(
    cmma: CmmaTile<E>,
    #[comptime] cfg: BounceConfig,
) -> Tile<E, Sc> {
    comptime!(assert_plane_scope(Sc::KIND));
    Tile::from_kind(TileKind::new_Bounce(BounceTile::<E>::new(cmma, cfg)))
}

#[cube]
impl<N: Numeric> BounceTile<N> {
    /// Copies into the bounce tile's cmma fragment from `source`. Bounce
    /// always loads through its CMMA representation (the WhiteboxFragment
    /// view is synced lazily on demand by softmax/scale ops); supported
    /// sources mirror [`CmmaTile::copy_from`].
    pub fn copy_from<SE: Numeric, SS: Size, Sc: TileScope>(
        &mut self,
        source: &Tile<SE, Sc>,
        #[comptime] ident: StageIdent,
    ) {
        self.cmma.copy_from::<SE, SS, Sc>(source, ident);
    }

    /// Zero-init the bounce tile (clears its cmma fragment).
    pub fn init_zero(&mut self) {
        self.cmma.init_zero();
    }
}

#[cube]
impl<Acc: Float> BounceTile<Acc> {
    /// Online softmax for the Bounce variant. cmma → fragment once at entry
    /// so all subsequent rowwise ops read/write the fragment view; the post-
    /// exp values are still in the fragment at the end (we skip
    /// `fragment_to_cmma` on `score` because its cmma is cleared next
    /// iteration), and we stream straight into `softmaxed` via
    /// `write_fragment_to`.
    pub fn softmax<Lhs: Float, M: Mask>(
        &mut self,
        mask: &M,
        softmaxed: &mut Tile<Lhs, Plane>,
        state: &mut (RowWise<Acc>, RowWise<Acc>),
        head_dim_factor: Acc,
    ) -> RowWise<Acc> {
        let num_rows = comptime!(state.0.num_rows);
        let mut max_buf = RowWise::<Acc>::new_min_value(num_rows);
        let mut sum_buf = RowWise::<Acc>::new_zero(num_rows);

        self.cmma_to_fragment();

        self.scale_and_mask::<M>(head_dim_factor, mask);
        self.row_max(&mut max_buf, &state.0);
        self.exp_diff(&max_buf);
        self.row_sum(&mut sum_buf);

        let exp_m_diff = state.0.exp_diff(&max_buf);
        let new_l = exp_m_diff.mul(&state.1).add(&sum_buf);

        self.write_fragment_to::<Lhs, Plane>(softmaxed);

        RowWise::copy_from(&mut state.0, &max_buf);
        RowWise::copy_from(&mut state.1, &new_l);

        exp_m_diff
    }
}
