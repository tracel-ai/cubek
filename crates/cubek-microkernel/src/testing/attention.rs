//! The attention microkernels on one work item: one batch, one head, every query one plane's
//! rows, every key attended, no mask.

use crate::attention::plane::{
    PlaneFragmentAttentionMicrokernel, PlaneFragmentAttentionMicrokernelSettings,
    plane_fragment_attention_microkernel,
};
use crate::attention::{
    AttentionMicrokernel, AttentionOperands, AttentionReading, AttentionVisibility,
    AttentionWorkItem, B, D, DV, G, KV, Q, S, VisibleKeys,
};
use cubecl::prelude::*;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{Launcher, Partitioning, Space, Tile, TileArg, TileArgLaunch};
use cubecl::{self as cubecl, cube};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};

/// The elements of a float attention: what the query, keys and values are stored in, what it
/// sums in and what it writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionElements {
    pub input: ElemType,
    pub sum: ElemType,
    pub out: ElemType,
}

/// One plane fragment attention microkernel ready to launch on one work item: the microkernel the
/// device picks at `settings`, over one plane's queries and `steps` stages of keys.
#[derive(Debug, Clone)]
pub struct PlaneFragmentAttentionLaunch {
    pub microkernel: PlaneFragmentAttentionMicrokernel,
    pub item: AttentionWorkItem,
    /// The queries and the keys.
    pub seq: [usize; 2],
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
        steps: usize,
    ) -> Result<Self, Refusal> {
        let item = AttentionWorkItem {
            elements: AttentionOperands {
                query: elements.input,
                key: elements.input,
                value: elements.input,
                out: elements.out,
            },
            sum: elements.sum,
            reading: AttentionReading::AsStored,
            page_size: None,
            group: 1,
            head_dim,
            val_dim,
        };
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
            seq: [microkernel.seq_q(), microkernel.seq_kv() * steps],
            microkernel,
            item,
            ring,
            plane_width,
        })
    }

    /// `out = softmax(scale · query · keyᵀ) · value`, `query` `[1, 1, 1, queries, head_dim]`,
    /// `key` `[1, 1, keys, head_dim]`, `value` `[1, 1, keys, val_dim]` and `out`
    /// `[1, 1, 1, queries, val_dim]`, all row-major (`[B, KV, G, Q, _]` and `[B, KV, S, _]`);
    /// `lengths` holds the one sequence's attended keys, as `u32`. Where `logsumexp` is given,
    /// `[1, 1, 1, queries]` in the sum's element, each query's `ln Σ exp(score)` lands there.
    ///
    /// # Errors
    ///
    /// Why the tile DSL cannot bind the launch.
    #[allow(clippy::too_many_arguments)]
    pub fn launch(
        &self,
        client: &Client,
        query: TensorBinding,
        key: TensorBinding,
        value: TensorBinding,
        out: TensorBinding,
        logsumexp: Option<TensorBinding>,
        lengths: TensorArg,
        scale: f32,
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
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let served = self.microkernel.served(item);
        let widths = AttentionOperands {
            query: 1,
            key: 1,
            value: 1,
            out: 1,
        };
        let delivery = self.microkernel.delivery(&widths, item);
        let bound = |tensor: TensorBinding, element, axes: &'static [_]| {
            launcher.arg(tensor, element).axes(axes).vectorize(1)
        };
        let query = bound(query, served.elements.query, &[B, KV, G, Q, D]).build()?;
        let key = bound(key, served.elements.key, &[B, KV, S, D])
            .delivery(delivery)
            .build()?;
        let value = bound(value, served.elements.value, &[B, KV, S, DV])
            .delivery(delivery)
            .build()?;
        let out = bound(out, item.elements.out, &[B, KV, G, Q, DV]).build()?;
        let logsumexp = logsumexp
            .map(|logsumexp| bound(logsumexp, item.sum, &[B, KV, G, Q]).build())
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
                Option::<TileArgLaunch>::None.into(),
                lengths,
                queries as u32,
                scale,
                launcher.partitioning_arg(),
                self.microkernel,
                self.ring,
                served.kernel_elements(),
            );
        }
        Ok(())
    }
}

/// The one plane running its queries through stages of its own over every key
/// ([`plane_fragment_attention_microkernel`]).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn plane_fragment_attention_kernel<EA: Float, ES: Float>(
    query: &TileArg<'_>,
    key: &TileArg<'_>,
    value: &TileArg<'_>,
    out: &TileArg<'_>,
    logsumexp: ComptimeOption<&TileArg<'static>>,
    mask: ComptimeOption<&TileArg<'static>>,
    len: &Tensor<u32>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneFragmentAttentionMicrokernel,
    #[comptime] ring: StageRing,
    #[define(EA, ES)] _elements: [ElemType; 2],
) {
    let mask = mask.map(|mask| mask.tile(&partitioning));
    let q = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);
    let o = out.tile(&partitioning);
    let lse: ComptimeOption<Tile> = logsumexp.map(|logsumexp| logsumexp.tile(&partitioning));

    for plane in &partitioning {
        let visible = VisibleKeys::new(len, &plane, queries, &mask, AttentionVisibility::Attended);
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
