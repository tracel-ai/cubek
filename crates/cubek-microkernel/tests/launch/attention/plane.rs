//! The plane attention microkernels on one work item: every query one plane's rows.

use cubecl::prelude::*;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::space::{ComputeScope, CubeOrder, PlaneOrder};
use cubecl::tile::{Launcher, Partitioning, Space, Tile, TileTensor, TileTensorLaunch};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::attention::plane::{
    PlaneFragmentAttentionMicrokernel, PlaneFragmentAttentionMicrokernelSettings,
    plane_fragment_attention_microkernel,
};
use cubek_microkernel::attention::{
    AttentionMicrokernel, AttentionVisibility, AttentionWorkItem, B, D, DV, G, KV, Q, S,
    VisibleKeys,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};

use super::{AttentionElements, AttentionLaunch, AttentionLines};
use crate::launch::Lines;

/// One plane fragment attention microkernel ready to launch on one work item: the microkernel the
/// device picks at `settings`, over one plane's queries and `steps` stages of keys, its operands
/// bound as `lines` says.
#[derive(Debug, Clone)]
pub struct PlaneFragmentAttentionLaunch {
    pub microkernel: PlaneFragmentAttentionMicrokernel,
    pub item: AttentionWorkItem,
    /// The queries and the keys.
    pub seq: [usize; 2],
    lines: AttentionLines,
    ring: StageRing,
    plane_width: u32,
}

impl PlaneFragmentAttentionLaunch {
    /// The best plane fragment attention microkernel `hw` runs at `settings`, with rows of
    /// `head_dim` in the query and key and of `val_dim` in the value, over `steps` stages of keys.
    ///
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(
        hw: &Hardware,
        elements: AttentionElements,
        [head_dim, val_dim]: [usize; 2],
        settings: &PlaneFragmentAttentionMicrokernelSettings,
        lines: Lines,
        steps: usize,
    ) -> Result<Self, Refusal> {
        let item = elements.work_item([head_dim, val_dim]);
        let microkernel = PlaneFragmentAttentionMicrokernel::best(hw, &item, settings)?;
        let plane_width = hw.plane_dim()?;
        let ring = StageRing::SINGLE.resolve(
            hw,
            &microkernel
                .served(&item)
                .stage(microkernel.seq_kv(), microkernel.key_order()),
            plane_width as usize,
        );
        Ok(Self {
            lines: AttentionLines::new(hw, lines, &item),
            seq: [microkernel.seq_q(), microkernel.seq_kv() * steps],
            microkernel,
            item,
            ring,
            plane_width,
        })
    }
}

impl AttentionLaunch for PlaneFragmentAttentionLaunch {
    fn item(&self) -> &AttentionWorkItem {
        &self.item
    }

    fn seq(&self) -> [usize; 2] {
        self.seq
    }

    fn launch(
        &self,
        client: &Client,
        query: TensorBinding,
        key: TensorBinding,
        value: TensorBinding,
        out: TensorBinding,
        logsumexp: Option<TensorBinding>,
        lengths: TensorArg,
        scale: f32,
        visibility: AttentionVisibility,
    ) -> Result<(), Refusal> {
        let [queries, keys] = self.seq;
        let item = &self.item;
        let space = Space::new(&[
            (B, 1),
            (KV, 1),
            (G, 1),
            (Q, queries),
            (S, keys),
            (D, item.head_dim),
            (DV, item.val_dim),
        ]);
        let partitioning = Partitioning::new(
            space.clone(),
            self.microkernel
                .tiling()
                .walk_every(&[S])
                .launch_planes(
                    &[Q, G, B, KV],
                    PlaneOrder::new(&[(Q, 1)], CubeOrder::RowMajor),
                )
                .build(),
        );
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(self.plane_width),
        };
        // A cube holds one plane, so its box is the plane's.
        let widths = self
            .lines
            .widths(&self.microkernel, &partitioning.box_of(ComputeScope::Plane));
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let served = self.microkernel.served(item);
        let delivery = self.microkernel.delivery(&widths, item);
        let bound = |tensor: TensorBinding, element, width, axes: &'static [_]| {
            launcher.arg(tensor, element).axes(axes).vectorize(width)
        };
        let query = bound(
            query,
            served.elements.query,
            widths.query,
            &[B, KV, G, Q, D],
        )
        .delivery(self.microkernel.query_delivery(&widths, item))
        .build()?;
        let key = bound(key, served.elements.key, widths.key, &[B, KV, S, D])
            .delivery(delivery)
            .build()?;
        let value = bound(value, served.elements.value, widths.value, &[B, KV, S, DV])
            .delivery(delivery)
            .build()?;
        let out = bound(out, item.elements.out, widths.out, &[B, KV, G, Q, DV]).build()?;
        // A value a query, written by whoever holds its row.
        let logsumexp = logsumexp
            .map(|logsumexp| bound(logsumexp, item.sum, 1, &[B, KV, G, Q]).build())
            .transpose()?;
        // SAFETY: every operand is bound through the launcher at the problem's own extents, the
        // keys and values are read no further than the attended length, and the kernel indexes
        // no buffer but through its tiles and the one entry of the lengths.
        unsafe {
            plane_fragment_attention_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                query.arg(),
                key.arg(),
                value.arg(),
                out.arg(),
                logsumexp.map(|logsumexp| logsumexp.arg()).into(),
                Option::<TileTensorLaunch>::None.into(),
                lengths,
                queries as u32,
                scale,
                launcher.partitioning_arg(),
                self.microkernel,
                self.ring,
                visibility,
                served.kernel_elements(),
            );
        }
        Ok(())
    }
}

/// The one plane running its queries through stages of its own over every key
/// ([`plane_fragment_attention_microkernel`]).
#[cube(launch_unchecked)]
fn plane_fragment_attention_kernel<EA: Float, ES: Float>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    out: &TileTensor<'_>,
    logsumexp: ComptimeOption<&TileTensor<'static>>,
    mask: ComptimeOption<&TileTensor<'static>>,
    len: &Tensor<u32>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneFragmentAttentionMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, ES)] _elements: [ElemType; 2],
) {
    let mask = mask.map(|mask| mask.tile(&partitioning));
    let q = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);
    let o = out.tile(&partitioning);
    let lse: ComptimeOption<Tile> = logsumexp.map(|logsumexp| logsumexp.tile(&partitioning));

    for plane in &partitioning {
        let visible = VisibleKeys::new(
            len,
            &ComptimeOption::new_None(),
            &plane,
            queries,
            &mask,
            visibility,
        );
        let k = key.within(S, 0, visible.attended as usize);
        let v = value.within(S, 0, visible.attended as usize);
        let bias = visible.bias::<EA>(&partitioning);
        let steps = visible.steps(&plane, microkernel.seq_q(), microkernel.seq_kv(), 1usize);

        plane_fragment_attention_microkernel::<EA, ES>(
            &plane,
            &o.at(&plane),
            &lse.clone().map(|lse| lse.at(&plane)),
            &q.at(&plane),
            &k.at(&plane),
            &v.at(&plane),
            &bias.at(&plane),
            steps,
            EA::cast_from(scale),
            microkernel,
            ring,
            1usize,
        );
    }
}
