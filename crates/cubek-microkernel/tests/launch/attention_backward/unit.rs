//! The unit gradient microkernels on one work item: one unit's queries of one head walked over
//! every key into their gradient, or one unit's keys walked over every query into theirs and their
//! values'. The cube's only unit runs it.

use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::tile::space::UnitCube;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::attention::{AttentionVisibility, B, G, KV, Q, S, VisibleKeys};
use cubek_microkernel::attention_backward::key_gradient::unit::{
    UnitKeyGradientMicrokernel, UnitKeyGradientMicrokernelSettings, unit_key_gradient_microkernel,
};
use cubek_microkernel::attention_backward::query_gradient::unit::{
    UnitQueryGradientMicrokernel, UnitQueryGradientMicrokernelSettings,
    unit_query_gradient_microkernel,
};
use cubek_microkernel::attention_backward::{AttentionBackwardOperand, AttentionBackwardWorkItem};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

use super::{
    AttentionBackwardLaunch, AttentionBackwardOperandArgs, AttentionBackwardWrites, OneSequence,
};

/// The one unit a launch here holds.
const ONE_UNIT: UnitCube = UnitCube {
    plane_width: 1,
    planes: 1,
};

/// One unit query gradient microkernel ready to launch on one unit's queries over `steps` of its
/// steps of keys.
#[derive(Debug, Clone, Copy)]
pub struct UnitQueryGradientLaunch {
    pub microkernel: UnitQueryGradientMicrokernel,
    pub item: AttentionBackwardWorkItem,
    pub seq: [usize; 2],
}

impl UnitQueryGradientLaunch {
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(
        hw: &Hardware,
        item: AttentionBackwardWorkItem,
        settings: &UnitQueryGradientMicrokernelSettings,
        steps: usize,
    ) -> Result<Self, Refusal> {
        let microkernel = UnitQueryGradientMicrokernel::best(hw, &item, settings)?;
        Ok(Self {
            seq: [microkernel.seq_q(), microkernel.seq_kv() * steps],
            microkernel,
            item,
        })
    }
}

impl AttentionBackwardLaunch for UnitQueryGradientLaunch {
    fn seq(&self) -> [usize; 2] {
        self.seq
    }

    fn writes(&self) -> AttentionBackwardWrites {
        AttentionBackwardWrites::Queries
    }

    fn launch(
        &self,
        client: &Client,
        args: AttentionBackwardOperandArgs,
        scale: f32,
    ) -> Result<(), Refusal> {
        use AttentionBackwardOperand as Operand;
        let sequence = OneSequence {
            item: self.item,
            seq: self.seq,
        };
        let levels = self
            .microkernel
            .tiling()
            .launch_units(&[Q, G, B, KV], ONE_UNIT)
            .build();
        let launcher = sequence.launcher(client, levels, 1)?;
        let [queries, keys] = self.seq;
        let bound = |arg, operand| sequence.bound(arg, operand);
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles.
        unsafe {
            unit_query_gradient_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                bound(args.query, Operand::Query),
                bound(args.key, Operand::Key),
                bound(args.value, Operand::Value),
                bound(args.out, Operand::Out),
                bound(args.out_gradient, Operand::OutGradient),
                bound(args.logsumexp, Operand::Logsumexp),
                bound(args.delta, Operand::Delta),
                bound(args.query_gradient, Operand::QueryGradient),
                queries as u32,
                keys as u32,
                scale,
                launcher.partitioning_arg(),
                self.microkernel,
                self.item.sum,
            );
        }
        Ok(())
    }
}

/// One unit key gradient microkernel ready to launch on one unit's keys over `steps` of its steps
/// of queries.
#[derive(Debug, Clone, Copy)]
pub struct UnitKeyGradientLaunch {
    pub microkernel: UnitKeyGradientMicrokernel,
    pub item: AttentionBackwardWorkItem,
    pub seq: [usize; 2],
}

impl UnitKeyGradientLaunch {
    /// # Errors
    ///
    /// Why the device runs none.
    pub fn new(
        hw: &Hardware,
        item: AttentionBackwardWorkItem,
        settings: &UnitKeyGradientMicrokernelSettings,
        steps: usize,
    ) -> Result<Self, Refusal> {
        let microkernel = UnitKeyGradientMicrokernel::best(hw, &item, settings)?;
        Ok(Self {
            seq: [microkernel.seq_q() * steps, microkernel.seq_kv()],
            microkernel,
            item,
        })
    }
}

impl AttentionBackwardLaunch for UnitKeyGradientLaunch {
    fn seq(&self) -> [usize; 2] {
        self.seq
    }

    fn writes(&self) -> AttentionBackwardWrites {
        AttentionBackwardWrites::Keys
    }

    fn launch(
        &self,
        client: &Client,
        args: AttentionBackwardOperandArgs,
        scale: f32,
    ) -> Result<(), Refusal> {
        use AttentionBackwardOperand as Operand;
        let sequence = OneSequence {
            item: self.item,
            seq: self.seq,
        };
        let levels = self
            .microkernel
            .tiling()
            .launch_units(&[S, B, KV], ONE_UNIT)
            .build();
        let launcher = sequence.launcher(client, levels, 1)?;
        let [queries, keys] = self.seq;
        let bound = |arg, operand| sequence.bound(arg, operand);
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles.
        unsafe {
            unit_key_gradient_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                bound(args.query, Operand::Query),
                bound(args.key, Operand::Key),
                bound(args.value, Operand::Value),
                bound(args.out_gradient, Operand::OutGradient),
                bound(args.logsumexp, Operand::Logsumexp),
                bound(args.delta, Operand::Delta),
                bound(args.key_gradient, Operand::KeyGradient),
                bound(args.value_gradient, Operand::ValueGradient),
                queries as u32,
                keys as u32,
                scale,
                launcher.partitioning_arg(),
                self.microkernel,
                self.item.sum,
            );
        }
        Ok(())
    }
}

/// The one unit's queries: their dot products, then their gradient over every key.
#[cube(launch_unchecked)]
fn unit_query_gradient_kernel<EA: Float>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    out: &TileTensor<'_>,
    out_gradient: &TileTensor<'_>,
    logsumexp: &TileTensor<'_>,
    delta: &TileTensor<'_>,
    query_gradient: &TileTensor<'_>,
    queries: u32,
    keys: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitQueryGradientMicrokernel,
    #[define(EA)] _sum: ElemType,
) {
    let (q, k, v) = (
        query.tile(&partitioning),
        key.tile(&partitioning),
        value.tile(&partitioning),
    );
    let (o, d_o) = (out.tile(&partitioning), out_gradient.tile(&partitioning));
    let (lse, delta) = (logsumexp.tile(&partitioning), delta.tile(&partitioning));
    let d_q = query_gradient.tile(&partitioning);
    let visible = VisibleKeys::of_keys(keys, queries, AttentionVisibility::Attended);
    let bias = visible.bias::<EA>(&partitioning);
    for unit in &partitioning {
        let steps = visible.steps(&unit, microkernel.seq_q(), microkernel.seq_kv(), 1usize);
        unit_query_gradient_microkernel::<EA>(
            &unit,
            &d_q.at(&unit),
            &delta.at(&unit),
            &q.at(&unit),
            &k.at(&unit),
            &v.at(&unit),
            &o.at(&unit),
            &d_o.at(&unit),
            &lse.at(&unit),
            &bias.at(&unit),
            steps,
            EA::cast_from(scale),
            microkernel,
        );
    }
}

/// The one unit's keys: their gradient and their values' over every query.
#[cube(launch_unchecked)]
fn unit_key_gradient_kernel<EA: Float>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    out_gradient: &TileTensor<'_>,
    logsumexp: &TileTensor<'_>,
    delta: &TileTensor<'_>,
    key_gradient: &TileTensor<'_>,
    value_gradient: &TileTensor<'_>,
    queries: u32,
    keys: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitKeyGradientMicrokernel,
    #[define(EA)] _sum: ElemType,
) {
    let (q, k, v) = (
        query.tile(&partitioning),
        key.tile(&partitioning),
        value.tile(&partitioning),
    );
    let d_o = out_gradient.tile(&partitioning);
    let (lse, delta) = (logsumexp.tile(&partitioning), delta.tile(&partitioning));
    let (d_k, d_v) = (
        key_gradient.tile(&partitioning),
        value_gradient.tile(&partitioning),
    );
    let visible = VisibleKeys::of_keys(keys, queries, AttentionVisibility::Attended);
    let bias = visible.keys_first_bias::<EA>(&partitioning);
    for unit in &partitioning {
        let first_step = visible.first_query_step(&unit, microkernel.seq_q());
        unit_key_gradient_microkernel::<EA>(
            &unit,
            &d_k.at(&unit),
            &d_v.at(&unit),
            &q.at(&unit),
            &k.at(&unit),
            &v.at(&unit),
            &d_o.at(&unit),
            &lse.at(&unit),
            &delta.at(&unit),
            &bias.at(&unit),
            first_step,
            EA::cast_from(scale),
            microkernel,
        );
    }
}
