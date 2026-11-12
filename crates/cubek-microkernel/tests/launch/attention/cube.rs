//! The cube attention microkernel on one cube's box: its planes over one box of queries, every key
//! attended. Planes of shuffles read each step's keys where they lie; planes of fragments contract
//! stages the cube fills.

use cubecl::prelude::*;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::space::ComputeScope;
use cubecl::tile::{
    Launcher, Partitioning, Space, StageStorage, Tile, TileTensor, TileTensorLaunch,
};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::attention::cube::{
    CubeAttentionMicrokernel, CubeAttentionMicrokernelSettings, CubeAttentionPlane,
    cube_staged_attention_microkernel, cube_unstaged_attention_microkernel,
};
use cubek_microkernel::attention::plane::TileStepReads;
use cubek_microkernel::attention::{
    AttentionMicrokernel, AttentionVisibility, AttentionWorkItem, B, D, DV, G, KV, Q, S,
    VisibleKeys,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;

use super::{AttentionElements, AttentionLaunch, AttentionLines};
use crate::launch::Lines;

/// One cube attention microkernel ready to launch on one cube's box of queries, `steps` of its
/// steps of keys, its operands bound as `lines` says.
#[derive(Debug, Clone)]
pub struct CubeAttentionLaunch {
    pub microkernel: CubeAttentionMicrokernel,
    pub item: AttentionWorkItem,
    pub seq: [usize; 2],
    lines: AttentionLines,
    storage: StageStorage,
    plane_width: u32,
}

impl CubeAttentionLaunch {
    /// The cube attention microkernel `hw` runs at `settings`, with rows of `head_dim` in the
    /// query and key and of `val_dim` in the value, over `steps` steps of keys.
    ///
    /// # Errors
    ///
    /// Why the device runs none, its planes cannot read its stages or merge their rows, or it
    /// cannot hold its stages in its shared memory.
    pub fn new(
        hw: &Hardware,
        elements: AttentionElements,
        dims: [usize; 2],
        settings: &CubeAttentionMicrokernelSettings,
        lines: Lines,
        steps: usize,
    ) -> Result<Self, Refusal> {
        let item = elements.work_item(dims);
        let microkernel = CubeAttentionMicrokernel::best(hw, &item, settings)?;
        let step_keys = match microkernel.plane {
            CubeAttentionPlane::Fragment(_) => microkernel.seq_kv(),
            CubeAttentionPlane::Shuffle(_) => microkernel.region_keys(),
        };
        let seq = [microkernel.seq_q(), step_keys * steps];
        let [head_dim, val_dim] = dims;
        let stage = Space::new(&[
            (B, 1),
            (KV, 1),
            (G, 1),
            (Q, microkernel.seq_q()),
            (S, microkernel.seq_kv()),
            (D, head_dim),
            (DV, val_dim),
        ]);
        let storage = microkernel.stage_storage(&stage);
        microkernel.plane.reads_stage(settings.layout)?;
        if settings.planes_along_keys.get() > 1 {
            microkernel.plane.merges_rows()?;
        }
        if microkernel.shared_memory(hw, &item, &storage) > hw.shared_memory_budget() {
            return Err(Refusal::Declined {
                why: "the cube's stages hold more shared memory than the device has",
            });
        }
        Ok(Self {
            lines: AttentionLines::new(hw, lines, &item),
            microkernel,
            item,
            seq,
            storage,
            plane_width: hw.plane_dim()?,
        })
    }
}

impl AttentionLaunch for CubeAttentionLaunch {
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
        let microkernel = self.microkernel;
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
            microkernel
                .tiling()
                .walk_every(&[S])
                .cubes(&[Q, G])
                .cubes(&[B, KV])
                .build(),
        );
        debug_assert_eq!(
            partitioning.box_of(ComputeScope::Cube).extent(Q),
            queries,
            "one cube holds every query"
        );
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(self.plane_width),
        };
        let widths = self
            .lines
            .widths(&microkernel, &partitioning.box_of(ComputeScope::Cube));
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let served = microkernel.served(item);
        let delivery = microkernel.delivery(&widths, item);
        let bound = |tensor: TensorBinding, element, width, axes: &'static [_]| {
            launcher.arg(tensor, element).axes(axes).vectorize(width)
        };
        let query = bound(
            query,
            served.elements.query,
            widths.query,
            &[B, KV, G, Q, D],
        )
        .delivery(microkernel.query_delivery(&widths, item))
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
        let logsumexp = logsumexp.map(|logsumexp| logsumexp.arg()).into();
        let no_mask = Option::<TileTensorLaunch>::None.into();
        // SAFETY: every operand is bound through the launcher at the problem's own extents, the
        // keys and values are read no further than the attended length, and the kernel indexes
        // no buffer but through its tiles and the one entry of the lengths; its shared memory is
        // the cube's own stages and the slots its planes merge in.
        unsafe {
            match microkernel.plane {
                CubeAttentionPlane::Fragment(_) => cube_staged_attention_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    query.arg(),
                    key.arg(),
                    value.arg(),
                    out.arg(),
                    logsumexp,
                    no_mask,
                    lengths,
                    queries as u32,
                    scale,
                    launcher.partitioning_arg(),
                    microkernel,
                    self.storage.clone(),
                    visibility,
                    served.kernel_elements(),
                ),
                CubeAttentionPlane::Shuffle(_) => cube_unstaged_attention_kernel::launch_unchecked(
                    client,
                    launcher.cube_count(),
                    launcher.cube_dim(),
                    // The plane shuffle lines every operand alike.
                    widths.query,
                    query.arg(),
                    key.arg(),
                    value.arg(),
                    out.arg(),
                    logsumexp,
                    no_mask,
                    lengths,
                    queries as u32,
                    scale,
                    launcher.partitioning_arg(),
                    microkernel,
                    item.val_dim,
                    visibility,
                    item.kernel_elements(),
                ),
            }
        }
        Ok(())
    }
}

/// The one cube walking the stages it fills over every key.
#[cube(launch_unchecked)]
fn cube_staged_attention_kernel<EA: Float, ES: Float>(
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
    #[comptime] microkernel: CubeAttentionMicrokernel,
    #[comptime] storage: StageStorage,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, ES)] _elements: [ElemType; 2],
) {
    let mask = mask.map(|mask| mask.tile(&partitioning));
    let q = query.tile(&partitioning);
    let key = key.tile(&partitioning);
    let value = value.tile(&partitioning);
    let o = out.tile(&partitioning);
    let lse: ComptimeOption<Tile> = logsumexp.map(|logsumexp| logsumexp.tile(&partitioning));
    for sequence in &partitioning {
        for cube in sequence {
            let visible = VisibleKeys::new(
                len,
                &ComptimeOption::new_None(),
                &cube,
                queries,
                &mask,
                visibility,
            );
            let k = key.within(S, 0, visible.attended as usize);
            let v = value.within(S, 0, visible.attended as usize);
            let bias = visible.step_bias::<EA>(&partitioning, &cube);
            let steps = visible.steps(&cube, microkernel.seq_q(), microkernel.seq_kv(), 1usize);
            cube_staged_attention_microkernel::<EA, ES>(
                &cube,
                &o.at(&cube),
                &lse.clone().map(|lse| lse.at(&cube)),
                &q.at(&cube),
                &k.at(&cube),
                &v.at(&cube),
                &bias.at(&cube),
                steps,
                EA::cast_from(scale),
                microkernel,
                storage.clone(),
            );
        }
    }
}

/// The one cube's planes of shuffles reading every step's keys where they lie.
#[cube(launch_unchecked)]
fn cube_unstaged_attention_kernel<EA: Float, E: Float, N: Size>(
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
    #[comptime] microkernel: CubeAttentionMicrokernel,
    #[comptime] value_dim: usize,
    #[comptime] visibility: AttentionVisibility,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let mask = mask.map(|mask| mask.tile(&partitioning));
    let q = query.tile(&partitioning);
    let k = TileStepReads::<E>::new(&key.tile(&partitioning));
    let v = TileStepReads::<E>::new(&value.tile(&partitioning));
    let o = out.tile(&partitioning);
    let lse: ComptimeOption<Tile> = logsumexp.map(|logsumexp| logsumexp.tile(&partitioning));
    for sequence in &partitioning {
        for cube in sequence {
            let visible = VisibleKeys::new(
                len,
                &ComptimeOption::new_None(),
                &cube,
                queries,
                &mask,
                visibility,
            );
            let steps = visible.steps(
                &cube,
                microkernel.seq_q(),
                microkernel.region_keys(),
                1usize,
            );
            cube_unstaged_attention_microkernel::<EA, E, E, N, TileStepReads<E>, TileStepReads<E>>(
                &cube,
                &o.at(&cube),
                &lse.clone().map(|lse| lse.at(&cube)),
                &q.at(&cube),
                &k,
                &v,
                &visible,
                steps,
                EA::cast_from(scale),
                value_dim,
                microkernel,
            );
        }
    }
}
