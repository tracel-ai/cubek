//! The hypercube key gradient microkernel on one work item: one cube's keys walked over every
//! query, a stage of queries at a time filled by all its planes, into their gradient and their
//! values', its share of the queries' gradient added into an `f32` sum.

use cubecl::features::AtomicUsage;
use cubecl::ir::Type;
use cubecl::prelude::*;
use cubecl::tile::launch::{AccumulateArgLaunch, Output, OutputArgs, Refusal};
use cubecl::tile::{Partitioning, TileSpec, TileTensor};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::attention::{AttentionVisibility, B, KV, Q, S, VisibleKeys};
use cubek_microkernel::attention_backward::key_gradient::cube::{
    QueryGradientShare, cube_staged_key_gradient_microkernel,
};
use cubek_microkernel::attention_backward::key_gradient::hypercube::{
    HypercubeKeyGradientMicrokernel, HypercubeKeyGradientMicrokernelSettings,
};
use cubek_microkernel::attention_backward::{AttentionBackwardOperand, AttentionBackwardWorkItem};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};

use super::{
    AttentionBackwardLaunch, AttentionBackwardOperandArgs, AttentionBackwardWrites, OneSequence,
};

/// One hypercube key gradient microkernel ready to launch on one cube's keys over `steps` of its
/// stages of queries.
#[derive(Debug, Clone, Copy)]
pub struct HypercubeKeyGradientLaunch {
    pub microkernel: HypercubeKeyGradientMicrokernel,
    pub item: AttentionBackwardWorkItem,
    pub seq: [usize; 2],
    ring: StageRing,
    plane_width: u32,
}

impl HypercubeKeyGradientLaunch {
    /// # Errors
    ///
    /// Why the device runs none: no `f32` atomic add for the sum, or no room for its stage.
    pub fn new(
        hw: &Hardware,
        item: AttentionBackwardWorkItem,
        settings: &HypercubeKeyGradientMicrokernelSettings,
        steps: usize,
    ) -> Result<Self, Refusal> {
        if !hw
            .properties
            .atomic_type_usage(Type::atomic(item.sum))
            .contains(AtomicUsage::Add)
        {
            return Err(Refusal::Declined {
                why: "the hypercube sums the query's gradient with atomic adds the device lacks",
            });
        }
        let microkernel = HypercubeKeyGradientMicrokernel::best(hw, &item, settings)?;
        let plane_width = hw.plane_dim()?;
        let seq = [microkernel.seq_q() * steps, microkernel.seq_kv()];
        let ring = OneSequence { item, seq }.ring(
            hw,
            Q,
            microkernel.seq_q(),
            microkernel.window_bytes(&item),
            microkernel.share_bytes(&item),
            plane_width * microkernel.plane_count() as u32,
        )?;
        Ok(Self {
            microkernel,
            item,
            seq,
            ring,
            plane_width,
        })
    }
}

impl AttentionBackwardLaunch for HypercubeKeyGradientLaunch {
    fn seq(&self) -> [usize; 2] {
        self.seq
    }

    fn writes(&self) -> AttentionBackwardWrites {
        AttentionBackwardWrites::KeysAndQuerySum
    }

    fn launch(
        &self,
        client: &Client,
        args: AttentionBackwardOperandArgs,
        scale: f32,
        visibility: AttentionVisibility,
    ) -> Result<(), Refusal> {
        use AttentionBackwardOperand as Operand;
        let sequence = OneSequence {
            item: self.item,
            seq: self.seq,
        };
        let levels = self
            .microkernel
            .tiling()
            .cubes(&[S])
            .batches(&[B, KV])
            .build();
        let launcher = sequence.launcher(client, levels, self.plane_width)?;
        let [queries, keys] = self.seq;
        let bound = |arg, operand| sequence.bound(arg, operand);
        let query_sum = OutputArgs::Atomic(AccumulateArgLaunch::new(
            args.query_gradient,
            1,
            TileSpec::direct(Operand::QueryGradient.axes()),
        ));
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles; it only adds into the query's sum,
        // atomically, and its shared memory is its stage and its share.
        unsafe {
            hypercube_key_gradient_kernel::launch_unchecked(
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
                query_sum,
                queries as u32,
                keys as u32,
                scale,
                launcher.partitioning_arg(),
                self.microkernel,
                self.ring,
                visibility,
                [self.item.sum, self.item.element],
            );
        }
        Ok(())
    }
}

/// The one cube's keys: their gradient and their values' over every query, and its share of the
/// queries' gradient added into `query_gradient`.
#[cube(launch_unchecked)]
fn hypercube_key_gradient_kernel<EA: Float, ES: Float>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    out_gradient: &TileTensor<'_>,
    logsumexp: &TileTensor<'_>,
    delta: &TileTensor<'_>,
    key_gradient: &TileTensor<'_>,
    value_gradient: &TileTensor<'_>,
    query_gradient: &Output<'_, f32>,
    queries: u32,
    keys: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: HypercubeKeyGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, ES)] _elements: [ElemType; 2],
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
    let d_q = query_gradient.tile(&partitioning);
    let visible = VisibleKeys::of_keys(keys, queries, visibility);
    let bias = visible.keys_first_bias::<EA>(&partitioning);
    for cube in &partitioning {
        let first_step = visible.first_query_step(&cube, microkernel.seq_q());
        let share = ComptimeOption::new_Some(QueryGradientShare {
            sum: d_q.at(&cube),
            product: microkernel.query_gradient,
        });
        cube_staged_key_gradient_microkernel::<EA, ES>(
            &cube,
            &d_k.at(&cube),
            &d_v.at(&cube),
            &share,
            &q.at(&cube),
            &k.at(&cube),
            &v.at(&cube),
            &d_o.at(&cube),
            &lse.at(&cube),
            &delta.at(&cube),
            &bias.at(&cube),
            first_step,
            EA::cast_from(scale),
            microkernel.cube,
            ring,
        );
    }
}
