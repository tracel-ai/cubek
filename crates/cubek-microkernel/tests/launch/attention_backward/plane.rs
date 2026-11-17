//! The plane gradient microkernels on one work item: one plane's queries of one head walked over
//! every key, a stage of keys at a time, into their gradient; or one plane's keys walked over every
//! query into theirs and their values'.

use cubecl::prelude::*;
use cubecl::tile::launch::Refusal;
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::attention::{AttentionVisibility, B, G, KV, Q, S, VisibleKeys};
use cubek_microkernel::attention_backward::key_gradient::plane::{
    PlaneKeyGradientMicrokernel, PlaneKeyGradientMicrokernelSettings,
    plane_fragment_key_gradient_microkernel,
};
use cubek_microkernel::attention_backward::query_gradient::plane::{
    PlaneQueryGradientMicrokernel, PlaneQueryGradientMicrokernelSettings,
    plane_fragment_query_gradient_microkernel,
};
use cubek_microkernel::attention_backward::{AttentionBackwardOperand, AttentionBackwardWorkItem};
use cubek_std::Hardware;
use cubek_std::microkernel::{Microkernel, StageRing};

use super::{
    AttentionBackwardLaunch, AttentionBackwardOperandArgs, AttentionBackwardWrites, OneSequence,
};

/// One plane query gradient microkernel ready to launch on one plane's queries over `steps` of
/// its stages of keys.
#[derive(Debug, Clone, Copy)]
pub struct PlaneQueryGradientLaunch {
    pub microkernel: PlaneQueryGradientMicrokernel,
    pub item: AttentionBackwardWorkItem,
    pub seq: [usize; 2],
    ring: StageRing,
    plane_width: u32,
}

impl PlaneQueryGradientLaunch {
    /// # Errors
    ///
    /// Why the device runs none, or cannot hold its stage.
    pub fn new(
        hw: &Hardware,
        item: AttentionBackwardWorkItem,
        settings: &PlaneQueryGradientMicrokernelSettings,
        steps: usize,
    ) -> Result<Self, Refusal> {
        let microkernel = PlaneQueryGradientMicrokernel::best(hw, &item, settings)?;
        let plane_width = hw.plane_dim()?;
        let seq = [microkernel.seq_q(), microkernel.seq_kv() * steps];
        let ring = OneSequence { item, seq }.ring(
            hw,
            S,
            microkernel.seq_kv(),
            microkernel.window_bytes(&item),
            0,
            plane_width,
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

impl AttentionBackwardLaunch for PlaneQueryGradientLaunch {
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
            .launch_planes(
                &[Q, G, B, KV],
                PlaneOrder::new(&[(Q, 1)], CubeOrder::RowMajor),
            )
            .build();
        let launcher = sequence.launcher(client, levels, self.plane_width)?;
        let [queries, keys] = self.seq;
        let bound = |arg, operand| sequence.bound(arg, operand);
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles; its shared memory is its stage.
        unsafe {
            plane_query_gradient_kernel::launch_unchecked(
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
                self.ring,
                visibility,
                [self.item.sum, self.item.element],
            );
        }
        Ok(())
    }
}

/// One plane key gradient microkernel ready to launch on one plane's keys over `steps` of its
/// stages of queries.
#[derive(Debug, Clone, Copy)]
pub struct PlaneKeyGradientLaunch {
    pub microkernel: PlaneKeyGradientMicrokernel,
    pub item: AttentionBackwardWorkItem,
    pub seq: [usize; 2],
    ring: StageRing,
    plane_width: u32,
}

impl PlaneKeyGradientLaunch {
    /// # Errors
    ///
    /// Why the device runs none, or cannot hold its stage.
    pub fn new(
        hw: &Hardware,
        item: AttentionBackwardWorkItem,
        settings: &PlaneKeyGradientMicrokernelSettings,
        steps: usize,
    ) -> Result<Self, Refusal> {
        let microkernel = PlaneKeyGradientMicrokernel::best(hw, &item, settings)?;
        let plane_width = hw.plane_dim()?;
        let seq = [microkernel.seq_q() * steps, microkernel.seq_kv()];
        let ring = OneSequence { item, seq }.ring(
            hw,
            Q,
            microkernel.seq_q(),
            microkernel.window_bytes(&item),
            0,
            plane_width,
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

impl AttentionBackwardLaunch for PlaneKeyGradientLaunch {
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
            .launch_planes(&[S, B, KV], PlaneOrder::new(&[(S, 1)], CubeOrder::RowMajor))
            .build();
        let launcher = sequence.launcher(client, levels, self.plane_width)?;
        let [queries, keys] = self.seq;
        let bound = |arg, operand| sequence.bound(arg, operand);
        // SAFETY: every operand is bound through the launcher at the problem's own extents, and
        // the kernel indexes no buffer but through its tiles; its shared memory is its stage.
        unsafe {
            plane_key_gradient_kernel::launch_unchecked(
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
                self.ring,
                visibility,
                [self.item.sum, self.item.element],
            );
        }
        Ok(())
    }
}

/// The one plane's queries: their dot products, then their gradient over every key.
#[cube(launch_unchecked)]
fn plane_query_gradient_kernel<EA: Float, ES: Float>(
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
    #[comptime] microkernel: PlaneQueryGradientMicrokernel,
    #[comptime] ring: StageRing,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, ES)] _elements: [ElemType; 2],
) {
    let (q, k, v) = (
        query.tile(&partitioning),
        key.tile(&partitioning),
        value.tile(&partitioning),
    );
    let (o, d_o) = (out.tile(&partitioning), out_gradient.tile(&partitioning));
    let (lse, delta) = (logsumexp.tile(&partitioning), delta.tile(&partitioning));
    let d_q = query_gradient.tile(&partitioning);
    let visible = VisibleKeys::of_keys(keys, queries, visibility);
    let bias = visible.bias::<EA>(&partitioning);
    for plane in &partitioning {
        let steps = visible.steps(&plane, microkernel.seq_q(), microkernel.seq_kv(), 1usize);
        plane_fragment_query_gradient_microkernel::<EA, ES>(
            &plane,
            &d_q.at(&plane),
            &delta.at(&plane),
            &q.at(&plane),
            &k.at(&plane),
            &v.at(&plane),
            &o.at(&plane),
            &d_o.at(&plane),
            &lse.at(&plane),
            &bias.at(&plane),
            steps,
            EA::cast_from(scale),
            microkernel,
            ring,
            1usize,
        );
    }
}

/// The one plane's keys: their gradient and their values' over every query.
#[cube(launch_unchecked)]
fn plane_key_gradient_kernel<EA: Float, ES: Float>(
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
    #[comptime] microkernel: PlaneKeyGradientMicrokernel,
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
    let visible = VisibleKeys::of_keys(keys, queries, visibility);
    let bias = visible.keys_first_bias::<EA>(&partitioning);
    for plane in &partitioning {
        let first_step = visible.first_query_step(&plane, microkernel.seq_q());
        plane_fragment_key_gradient_microkernel::<EA, ES>(
            &plane,
            &d_k.at(&plane),
            &d_v.at(&plane),
            &q.at(&plane),
            &k.at(&plane),
            &v.at(&plane),
            &d_o.at(&plane),
            &lse.at(&plane),
            &delta.at(&plane),
            &bias.at(&plane),
            first_step,
            EA::cast_from(scale),
            microkernel,
            ring,
            1usize,
        );
    }
}
