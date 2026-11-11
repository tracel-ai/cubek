//! The hypercube attention microkernel on one box of queries whose keys its cubes share: each cube
//! walks its run of the keys, its planes leaving their partials in its slot, and the last cube of
//! the box to arrive merges the slots into the output and the logsumexp.

use core::marker::PhantomData;
use core::num::NonZeroUsize;
use cubecl::prelude::*;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{Grid, Refusal};
use cubecl::tile::{Launcher, Partitioning, Space, StageStorage, TileTensor};
use cubecl::{self as cubecl, cube};
use cubek_microkernel::attention::cube::{
    CubeAttentionPlane, cube_unstaged_attention_partial_microkernel,
};
use cubek_microkernel::attention::hypercube::{
    HypercubeAttentionMicrokernel, HypercubeAttentionMicrokernelSettings,
};
use cubek_microkernel::attention::plane::TileStepReads;
use cubek_microkernel::attention::{
    AttentionMicrokernel, AttentionOperands, AttentionVisibility, AttentionWorkItem, B, D, DV, G,
    KV, Q, S, VisibleKeys,
};
use cubek_microkernel::split_keys_merge::hypercube::{
    HypercubeMergeShape, hypercube_split_keys_merge_microkernel,
};
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::TestInput;

use super::{AttentionElements, AttentionLaunch};

/// The cubes sharing the box's keys: three, so keys three steps long are cut into three runs whose
/// partials the last arrival merges, and keys one step long leave two cubes nothing to walk.
const CUBES: NonZeroUsize = NonZeroUsize::new(3).unwrap();

/// One hypercube attention microkernel ready to launch on one box of queries, `steps` of its cube's
/// steps of keys, the keys shared by [`CUBES`] cubes.
#[derive(Debug, Clone)]
pub struct HypercubeAttentionLaunch {
    pub microkernel: HypercubeAttentionMicrokernel,
    pub item: AttentionWorkItem,
    pub seq: [usize; 2],
    plane_width: u32,
}

impl HypercubeAttentionLaunch {
    /// The hypercube attention microkernel `hw` runs at `settings`, with rows of `head_dim` in the
    /// query and key and of `val_dim` in the value, over `steps` steps of keys.
    ///
    /// # Errors
    ///
    /// Why the device runs none: no cube microkernel at `settings`, planes of fragments, whose
    /// slots a second launch merges, more shared memory than the device has, or cubes whose
    /// stores do not reach one another within a launch.
    pub fn new(
        hw: &Hardware,
        elements: AttentionElements,
        dims: [usize; 2],
        settings: &HypercubeAttentionMicrokernelSettings,
        steps: usize,
    ) -> Result<Self, Refusal> {
        if !hw.properties.features.device_memory_scope {
            return Err(Refusal::Declined {
                why: "the last arrival reads what other cubes stored within the launch",
            });
        }
        let item = elements.work_item(dims);
        let microkernel = HypercubeAttentionMicrokernel::best(hw, &item, settings)?;
        if let CubeAttentionPlane::Fragment(_) = microkernel.cube.plane {
            return Err(Refusal::Declined {
                why: "planes of fragments leave slots a second launch merges, not the last arrival",
            });
        }
        let [head_dim, val_dim] = dims;
        let stage = Space::new(&[
            (B, 1),
            (KV, 1),
            (G, 1),
            (Q, microkernel.cube.seq_q()),
            (S, microkernel.cube.seq_kv()),
            (D, head_dim),
            (DV, val_dim),
        ]);
        let storage: StageStorage = microkernel.cube.stage_storage(&stage);
        // The word a cube learns it came last in, beside its walk's.
        let shared_memory = microkernel.shared_memory(hw, &item, &storage) + size_of::<u32>();
        if shared_memory > hw.shared_memory_budget() {
            return Err(Refusal::Declined {
                why: "the cube's walk holds more shared memory than the device has",
            });
        }
        Ok(Self {
            seq: [
                microkernel.cube.seq_q(),
                microkernel.cube.region_keys() * steps,
            ],
            microkernel,
            item,
            plane_width: hw.plane_dim()?,
        })
    }

    /// The merge the last arrival runs: a slot for every cube, a box of the cube's queries.
    fn merge(&self) -> HypercubeMergeShape {
        self.microkernel
            .merge_shape(&self.item, CUBES, self.microkernel.cube.seq_q())
    }
}

impl AttentionLaunch for HypercubeAttentionLaunch {
    fn item(&self) -> &AttentionWorkItem {
        &self.item
    }

    fn seq(&self) -> [usize; 2] {
        self.seq
    }

    /// Twice on the same arrival counters, first on zeros of its own into outputs of its own,
    /// then on the given operands: each box's last arrival must leave its counter at zero, or the
    /// second launch merges early or never.
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
        let arrivals = TensorHandle::zeros(
            client,
            [queries.div_ceil(self.merge().seq_q)],
            u32::elem_type_native(),
        );
        let served = self.microkernel.cube.served(item);
        let zeros = |dims: &[usize], element| TensorHandle::zeros(client, dims.to_vec(), element);
        let scratch = AttentionBindings {
            query: zeros(&[1, 1, 1, queries, item.head_dim], served.elements.query).binding(),
            key: zeros(&[1, 1, keys, item.head_dim], served.elements.key).binding(),
            value: zeros(&[1, 1, keys, item.val_dim], served.elements.value).binding(),
            out: zeros(&[1, 1, 1, queries, item.val_dim], item.elements.out).binding(),
            logsumexp: logsumexp
                .is_some()
                .then(|| zeros(&[1, 1, 1, queries], item.sum).binding()),
            lengths: TensorHandle::new_contiguous(
                [1],
                client.create_from_slice(u32::as_bytes(&[keys as u32])),
                u32::elem_type_native(),
            )
            .into_arg(),
        };
        self.dispatch(client, scratch, scale, &arrivals)?;
        let given = AttentionBindings {
            query,
            key,
            value,
            out,
            logsumexp,
            lengths,
        };
        self.dispatch(client, given, scale, &arrivals)
    }
}

/// What one launch reads and writes: the operands, the outputs the box's last cube writes, and
/// the attended length.
struct AttentionBindings {
    query: TensorBinding,
    key: TensorBinding,
    value: TensorBinding,
    out: TensorBinding,
    logsumexp: Option<TensorBinding>,
    lengths: TensorArg,
}

impl HypercubeAttentionLaunch {
    /// The kernel on `bindings`, the boxes counting in at `arrivals`, the slots holding NaN until
    /// a cube stores its partial.
    fn dispatch(
        &self,
        client: &Client,
        bindings: AttentionBindings,
        scale: f32,
        arrivals: &TensorHandle,
    ) -> Result<(), Refusal> {
        let AttentionBindings {
            query,
            key,
            value,
            out,
            logsumexp,
            lengths,
        } = bindings;
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
        let tiling = microkernel.cube.tiling();
        let walked = match microkernel.cube.page_steps {
            Some(steps) => tiling.walk(&[(S, steps.get())]),
            None => tiling,
        };
        let levels = walked
            .cubes(&[Q, G, S])
            .across(S, CUBES.get())
            .interleaved(S)
            .batches(&[B, KV])
            .build();
        let partitioning = Partitioning::new(space.clone(), levels);
        let grid = Grid::Stated {
            cube_count: partitioning.cube_count(),
            cube_dim: partitioning.cube_dim(self.plane_width),
        };
        let launcher = Launcher::new(client, partitioning, &space, grid)?;
        let served = microkernel.cube.served(item);
        let widths = AttentionOperands {
            query: 1,
            key: 1,
            value: 1,
            out: 1,
        };
        let delivery = AttentionMicrokernel::delivery(&microkernel.cube, &widths, item);
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
        let merge = self.merge();
        let slots = [1, 1, 1, queries, merge.slots.splits];
        let poisoned = |dims: Vec<usize>| {
            let cells = dims.iter().product();
            TestInput::builder(client.clone(), dims)
                .dtype(item.sum)
                .custom(vec![f32::NAN; cells])
                .generate_without_host_data()
        };
        let slot_out = poisoned([slots.as_slice(), &[item.val_dim]].concat());
        let slot_max = poisoned(slots.to_vec());
        let slot_sum = poisoned(slots.to_vec());
        // SAFETY: every operand the cubes read is bound through the launcher at the problem's own
        // extents, the keys and values read no further than the attended length; each cube writes
        // its own slots, counts itself in at its box's counter, which arrives holding zero, and the
        // box's last writes the box's queries of the output and the logsumexp at their strides.
        unsafe {
            hypercube_attention_kernel::launch_unchecked(
                client,
                launcher.cube_count(),
                launcher.cube_dim(),
                1,
                query.arg(),
                key.arg(),
                value.arg(),
                out.into_tensor_arg(),
                logsumexp
                    .map(|logsumexp| logsumexp.into_tensor_arg())
                    .into(),
                slot_out.into_arg(),
                slot_max.into_arg(),
                slot_sum.into_arg(),
                arrivals.clone().into_arg().into_buffer_arg(),
                lengths,
                queries as u32,
                scale,
                launcher.partitioning_arg(),
                microkernel,
                merge,
                item.kernel_elements(),
            );
        }
        Ok(())
    }
}

/// Every cube walking its run of the keys into its slot, the box's last arrival merging the slots.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn hypercube_attention_kernel<EA: Float, E: Float, N: Size>(
    query: &TileTensor<'_>,
    key: &TileTensor<'_>,
    value: &TileTensor<'_>,
    out: &mut Tensor<E>,
    logsumexp: ComptimeOption<&mut Tensor<EA>>,
    slot_out: &mut Tensor<Vector<EA, N>>,
    slot_max: &mut Tensor<EA>,
    slot_sum: &mut Tensor<EA>,
    arrivals: &[Atomic<u32>],
    len: &Tensor<u32>,
    queries: u32,
    scale: f32,
    partitioning: Partitioning,
    #[comptime] microkernel: HypercubeAttentionMicrokernel,
    #[comptime] merge: HypercubeMergeShape,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let key = TileStepReads::<E> {
        tile: key.tile(&partitioning),
        element: PhantomData,
    };
    let value = TileStepReads::<E> {
        tile: value.tile(&partitioning),
        element: PhantomData,
    };
    // The cube level's walk: this cube's run of the steps of keys.
    let run = partitioning.walk();
    let first_step = run.region(0);
    let q = query.tile(&partitioning);
    let visible = VisibleKeys::new(
        len,
        &ComptimeOption::new_None(),
        &first_step,
        queries,
        &ComptimeOption::new_None(),
        AttentionVisibility::Attended,
    );
    let steps = visible.steps(
        &first_step,
        microkernel.cube.seq_q(),
        microkernel.cube.region_keys(),
        merge.cubes,
    );
    let slot_box = cube_unstaged_attention_partial_microkernel::<
        EA,
        E,
        N,
        TileStepReads<E>,
        TileStepReads<E>,
    >(
        run,
        &q.at(&first_step),
        &key,
        &value,
        &visible,
        steps,
        EA::cast_from(scale),
        slot_out,
        slot_max,
        slot_sum,
        microkernel.cube,
        merge.slots.splits,
        merge.slots.val_dim,
    );
    hypercube_split_keys_merge_microkernel::<EA, E, N>(
        &slot_box, slot_out, slot_max, slot_sum, out, logsumexp, arrivals, merge,
    );
}
