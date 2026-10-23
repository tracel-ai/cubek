//! The grouped product's autotune, `out[r] = lhs[r] @ stacked[group(r)]`: the dense plane shuffle,
//! which holds a chunk of a group's rows against one read of its slab, raced against the dense
//! cube staged routine, which stages them, each launched on the groups padded to its own chunks
//! ([`MGroupedMatmulTensors`]). A packed stack is read packed. Any rows grouped by the slab they
//! read are one such product, not only an expert layer's.
//!
//! Its gradient for the stack is the K-grouped product, [`k_grouped`](crate::kernels::k_grouped).

use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tune::{LocalTuner, Tunable, local_tuner};
use cubecl::{self as cubecl, AutotuneKey, CubeTuneId};
use cubecl_common::device::Device;
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::Hardware;
use cubek_std::autotune::{Autotune, GroupPriority};
use serde::{Deserialize, Serialize};

use crate::kernels::grouped_rows::{GroupedRows, PaddedRows, RouteSort, upload};
use crate::kernels::m_grouped::{
    MGroupedMatmulTensors, MGroupedRowSource, MGroupedRowSourceKind, MGroupedSettings,
};
use crate::kernels::select::problem::OperandKind;
use crate::kernels::stack::SlabAxis;
use crate::kernels::tensor::PlainTensor;
use crate::kernels::tune::key::OperandLayout;
use crate::kernels::{
    CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings, MatmulRoutine,
    PlaneShuffleMatmulRoutine, PlaneShuffleMatmulRoutineSettings,
};

/// What makes two grouped products want the same launch: the stack's shape and layout, as
/// the routed key states them, and the rows, whose count over the groups' is the mean group's
/// size, which decides between holding a group's rows against one read of its slab and staging
/// them.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct MGroupedMatmulKey {
    /// The lhs, the stack and the output all bind at this element.
    elem: ElemType,
    /// The contraction.
    k: usize,
    /// One slab's output columns.
    n: usize,
    /// Slabs in the stack, one per group.
    groups: usize,
    /// Grouped rows, anchored to the power of two above them.
    #[autotune(anchor(exp(min = 1, max = 65536, base = 2)))]
    m: usize,
    /// Whether the stack is float or packed, and how: a packed stack is read packed, by other
    /// kernels than a float one.
    stack: OperandKind,
    /// Which axis the stack lines along.
    lines: OperandLayout,
    /// Which axis the slabs lie along.
    slabs: SlabAxis,
    /// Where the rows come from: a candidate is timed with the pass that pads them.
    source: MGroupedRowSourceKind,
}

/// The floors' settings, named once because the entry asks them of a problem too: a round must
/// reach a candidate that answers. Each declines where another runs: the plane shuffle where the
/// device commits to no plane width, its default against a packed stack, which its lead against
/// a packed rhs reads, and the cube staged routine on a CPU runtime, which holds fewer units a
/// cube than its planes, and against a packed stack.
const PLANE_SHUFFLE_FLOOR: MGroupedSettings<PlaneShuffleMatmulRoutineSettings> =
    MGroupedSettings::PLANE_SHUFFLE;
const PLANE_SHUFFLE_PACKED_FLOOR: MGroupedSettings<PlaneShuffleMatmulRoutineSettings> =
    MGroupedSettings::PLANE_SHUFFLE_PACKED;
const CUBE_STAGED_FLOOR: MGroupedSettings<CubeStagedMatmulRoutineSettings> =
    MGroupedSettings::CUBE_STAGED;

/// The grouped product `tensors` states, through whichever setting measures fastest for it on
/// `device`, unpadded: `[rows, n]` at the rows' element. `routines` is the priority of the
/// routines' group; the floors sit under it.
///
/// # Errors
///
/// As [`launch_tuned_padded`].
pub fn launch_tuned(
    device: &impl Device,
    tensors: MGroupedMatmulTensors,
    routines: GroupPriority,
) -> Result<TensorHandle, Refusal> {
    let rows = tensors.source().rows();
    if rows == 0 {
        let [_, n] = tensors.slab_shape();
        let elem = tensors.source().buffer().elem;
        return Ok(PlainTensor::empty(tensors.client(), [0, n].into(), elem).0);
    }
    launch_tuned_padded(device, tensors, routines).map(PaddedRows::unpad)
}

/// [`launch_tuned`], the output handed back padded to the winner's chunks, for a next step that
/// reads it padded.
///
/// # Errors
///
/// A product neither floor serves, a packed stack on a device whose plane shuffle declines it
/// among them, so no candidate of the autotune is known to: the caller names what runs instead.
///
/// # Panics
///
/// On no row.
pub fn launch_tuned_padded(
    device: &impl Device,
    tensors: MGroupedMatmulTensors,
    routines: GroupPriority,
) -> Result<PaddedRows, Refusal> {
    let client = tensors.client().clone();
    let hw = Hardware::from_client(&client);
    // A round must reach a floor that answers: the plane shuffle declines a device that commits
    // to no plane width, the cube staged routine a CPU runtime, which holds fewer units a cube
    // than its planes, and a packed stack.
    let cube_staged_served = tensors.served_by(&hw, &CubeStagedMatmulRoutine, &CUBE_STAGED_FLOOR);
    let packed_served =
        tensors.served_by(&hw, &PlaneShuffleMatmulRoutine, &PLANE_SHUFFLE_PACKED_FLOOR);
    tensors
        .served_by(&hw, &PlaneShuffleMatmulRoutine, &PLANE_SHUFFLE_FLOOR)
        .or(packed_served)
        .or(cube_staged_served)?;

    // Named rather than `module_path!()`, so the cached winners stay this autotune's wherever the
    // code lives; a cached winner is an index into the candidate list.
    static TUNER: LocalTuner<MGroupedMatmulKey, CubeTuneId> = local_tuner!("m-grouped");

    let id = CubeTuneId::new(&client, device);
    let tunables = TUNER.init(&id, move || {
        let mut autotune =
            Autotune::measured_on(MGroupedMatmulTensors::key, "evenly-spread", evenly_spread);
        let microkernel = autotune.group("microkernel", routines);
        let autotune = MGroupedSettings::plane_shuffle(&hw).into_iter().fold(
            autotune,
            |autotune, settings| {
                autotune.candidate(&microkernel, candidate(PlaneShuffleMatmulRoutine, settings))
            },
        );
        MGroupedSettings::cube_staged(&hw)
            .into_iter()
            .fold(autotune, |autotune, settings| {
                autotune.candidate(&microkernel, candidate(CubeStagedMatmulRoutine, settings))
            })
            .floor(candidate(PlaneShuffleMatmulRoutine, PLANE_SHUFFLE_FLOOR))
            .floor(candidate(
                PlaneShuffleMatmulRoutine,
                PLANE_SHUFFLE_PACKED_FLOOR,
            ))
            .floor(candidate(CubeStagedMatmulRoutine, CUBE_STAGED_FLOOR))
            .finish()
    });

    Ok(TUNER.execute(&id, &client, tunables, tensors))
}

/// `routine` at `settings`, launched on the grouped product, under the setting's name.
fn candidate<Mk, SM, R>(
    routine: R,
    settings: MGroupedSettings<R::Settings>,
) -> Tunable<MGroupedMatmulKey, MGroupedMatmulTensors, PaddedRows>
where
    Mk: MatmulMicrokernel,
    R: MatmulRoutine<Mk, SM> + Send + Sync + 'static,
    R::Settings: Send + Sync + 'static,
{
    Tunable::new(
        &settings.to_string(),
        move |tensors: MGroupedMatmulTensors| {
            let hw = Hardware::from_client(tensors.client());
            tensors
                .launch(&hw, &routine, &settings)
                .map_err(|reason| format!("{settings} does not serve this problem: {reason:?}"))
        },
    )
}

/// The operands a candidate is measured on: the call's rows spread evenly over the key's groups.
///
/// The environment build discards the sort that makes the call's `ends` and route places, so
/// they hold whatever the buffers did, and a kernel reading them as a layout runs on garbage.
/// Uploaded rather than launched, so a process mode that discards launches keeps them.
fn evenly_spread(
    key: &MGroupedMatmulKey,
    tensors: &MGroupedMatmulTensors,
) -> MGroupedMatmulTensors {
    let client = tensors.client();
    let source = tensors.source();
    let rows = source.rows();
    let spread: Vec<u32> = (1..=key.groups)
        .map(|group| (group * rows / key.groups) as u32)
        .collect();
    let ends = PlainTensor::of_handle(upload(client, &spread));
    let source = match source {
        MGroupedRowSource::Sorted(rows) => MGroupedRowSource::Sorted(rows.clone()),
        MGroupedRowSource::Routed { tokens, sort } => {
            let places: Vec<u32> = (0..rows as u32).collect();
            MGroupedRowSource::Routed {
                tokens: tokens.clone(),
                sort: RouteSort {
                    route_at_place: PlainTensor::of_handle(upload(client, &places)),
                    slots: sort.slots,
                },
            }
        }
        MGroupedRowSource::Gated(both) => {
            let align = both.layout().align();
            let layout = GroupedRows::uploaded(client, &spread, rows, align);
            MGroupedRowSource::Gated(PaddedRows::new(both.rows().clone(), layout))
        }
    };
    tensors.with_rows(source, ends)
}
