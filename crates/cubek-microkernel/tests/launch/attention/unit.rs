//! The unit attention microkernel on one work item: one unit's query rows against every key, the
//! cube's only unit.

use cubecl::prelude::*;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::space::{ComputeScope, UnitCube};
use cubecl::tile::{Launcher, Partitioning, Space, Tile, TileTensor, TileTensorLaunch};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::attention::unit::{
    UnitAttentionMicrokernel, UnitAttentionMicrokernelSettings, unit_attention_microkernel,
};
use cubek_microkernel::attention::{
    AttentionVisibility, AttentionWorkItem, B, D, DV, G, KV, Q, S, VisibleKeys,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

use super::{AttentionElements, AttentionLaunch, AttentionLines};
use crate::launch::Lines;

/// One unit attention microkernel ready to launch on one unit's rows, `steps` steps of keys, its
/// operands bound as `lines` says.
#[derive(Debug, Clone)]
pub struct UnitAttentionLaunch {
    pub microkernel: UnitAttentionMicrokernel,
    pub item: AttentionWorkItem,
    pub seq: [usize; 2],
    lines: AttentionLines,
}

impl UnitAttentionLaunch {
    /// The unit attention microkernel `hw` runs at `settings`, with rows of `head_dim` in the
    /// query and key and of `val_dim` in the value, over `steps` steps of keys.
    ///
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(
        hw: &Hardware,
        elements: AttentionElements,
        dims: [usize; 2],
        settings: &UnitAttentionMicrokernelSettings,
        lines: Lines,
        steps: usize,
    ) -> Result<Self, Refusal> {
        let item = elements.work_item(dims);
        let microkernel = UnitAttentionMicrokernel::best(hw, &item, settings)?;
        Ok(Self {
            lines: AttentionLines::new(hw, lines, &item),
            seq: [
                microkernel.settings.seq_q.get(),
                microkernel.seq_kv() * steps,
            ],
            microkernel,
            item,
        })
    }
}

impl AttentionLaunch for UnitAttentionLaunch {
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
                .launch_units(
                    &[Q, G, B, KV],
                    UnitCube {
                        plane_width: 1,
                        planes: 1,
                    },
                )
                .build(),
        );
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(1),
        };
        // A cube holds one unit, so its box is the unit's.
        let widths = self
            .lines
            .widths(&self.microkernel, &partitioning.box_of(ComputeScope::Unit));
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let bound = |tensor: TensorBinding, element, width, axes: &'static [_]| {
            launcher
                .arg(tensor, element)
                .axes(axes)
                .vectorize(width)
                .build()
        };
        let elements = item.elements;
        let query = bound(query, elements.query, widths.query, &[B, KV, G, Q, D])?;
        let key = bound(key, elements.key, widths.key, &[B, KV, S, D])?;
        let value = bound(value, elements.value, widths.value, &[B, KV, S, DV])?;
        let out = bound(out, elements.out, widths.out, &[B, KV, G, Q, DV])?;
        // A value a query, written by whoever holds its row.
        let logsumexp = logsumexp
            .map(|logsumexp| bound(logsumexp, item.sum, 1, &[B, KV, G, Q]))
            .transpose()?;
        // SAFETY: every operand is bound through the launcher at the problem's own extents, the
        // keys and values are read no further than the attended length, and the kernel indexes
        // no buffer but through its tiles and the one entry of the lengths.
        unsafe {
            unit_attention_kernel::launch_unchecked(
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
                item.sum,
            );
        }
        Ok(())
    }
}

/// The one unit running its rows over every key.
#[cube(launch_unchecked)]
fn unit_attention_kernel<EA: Float>(
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
    #[comptime] microkernel: UnitAttentionMicrokernel,
    #[define(EA)] _sum: ElemType,
) {
    let mask = mask.map(|mask| mask.tile(&partitioning));
    let q = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);
    let o = out.tile(&partitioning);
    let lse: ComptimeOption<Tile> = logsumexp.map(|logsumexp| logsumexp.tile(&partitioning));
    for unit in &partitioning {
        let visible = VisibleKeys::new(
            len,
            &ComptimeOption::new_None(),
            &unit,
            queries,
            &mask,
            AttentionVisibility::Attended,
        );
        let k = key.within(S, 0, visible.attended as usize);
        let v = value.within(S, 0, visible.attended as usize);
        let bias = visible.bias::<EA>(&partitioning);
        let steps = visible.steps(
            &unit,
            microkernel.settings.seq_q.get(),
            microkernel.seq_kv(),
            1usize,
        );
        unit_attention_microkernel::<EA>(
            &unit,
            &o.at(&unit),
            &lse.clone().map(|lse| lse.at(&unit)),
            &q.at(&unit),
            &k.at(&unit),
            &v.at(&unit),
            &bias.at(&unit),
            steps,
            EA::cast_from(scale),
            microkernel,
        );
    }
}
