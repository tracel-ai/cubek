//! The routed product's autotune, `out[r, s] = lhs[r] @ stacked[routes[r, s]]`: the dense plane
//! shuffle, cube staged and unit registers routines, each at the settings listed for the stack's
//! kind ([`Listing`]), raced on the product's own operands ([`RoutedMatmulTensors`]). A
//! packed stack is read packed.

use cubecl::ir::{ElemType, UIntKind};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tune::{LocalTuner, Tunable, TuneGroup, local_tuner};
use cubecl::{self as cubecl, AutotuneKey, CubeTuneId};
use cubecl_common::device::Device;
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::Hardware;
use cubek_std::autotune::{Autotune, GroupPriority};
use serde::{Deserialize, Serialize};

use crate::kernels::indexed::routed::{RoutedCandidates, RoutedFloors, RoutedMatmulTensors};
use crate::kernels::indexed::stack::SlabAxis;
use crate::kernels::select::problem::OperandKind;
use crate::kernels::tensor::PlainTensor;
use crate::kernels::tune::key::OperandLayout;
use crate::kernels::tune::{ListedFor, Listing, MatmulProduct};
use crate::kernels::{
    CubeStagedMatmulRoutine, MatmulRoutine, PlaneShuffleMatmulRoutine, UnitRegistersMatmulRoutine,
};

/// What makes two routed products want the same launch: the stack's shape and layout, the rows,
/// and the slots each row selects.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct RoutedMatmulKey {
    /// The lhs, the stack and the output all bind at this element.
    elem: ElemType,
    /// The contraction.
    k: usize,
    /// One slab's output columns.
    n: usize,
    /// Slabs in the stack.
    experts: usize,
    /// The lhs's rows, anchored to the power of two above them.
    rows: usize,
    /// Experts each row selects.
    slots: usize,
    /// Whether the stack is float or packed, and how: a packed stack is read packed, by other
    /// kernels than a float one.
    stack: OperandKind,
    /// Which axis the stack lines along.
    lines: OperandLayout,
    /// Which axis the slabs lie along.
    slabs: SlabAxis,
}

/// The routed product `tensors` states, through whichever setting measures fastest for it on
/// `device`: `[rows, slots, n]` at the rows' element. `routines` is the priority of the routines'
/// group; the floors sit under it.
///
/// # Errors
///
/// A product no floor serves, so no candidate of the autotune is known to: the caller names what
/// runs instead.
pub fn launch_tuned(
    device: &impl Device,
    tensors: RoutedMatmulTensors,
    routines: GroupPriority,
) -> Result<TensorHandle, Refusal> {
    if tensors.rows() == 0 {
        return Ok(tensors.output());
    }
    let client = tensors.client().clone();
    let hw = Hardware::from_client(&client);
    // A round must reach a floor that answers, and only a floor listed for the stack's kind can.
    let stack = tensors.product();
    let floors = RoutedFloors::LISTED;
    floors.served(&hw, stack, &tensors)?;

    // Named rather than `module_path!()`, so the cached winners stay this autotune's wherever the
    // code lives; a cached winner is an index into the candidate list.
    static TUNER: LocalTuner<RoutedMatmulKey, CubeTuneId> = local_tuner!("routed");

    let id = CubeTuneId::new(&client, device);
    let tunables = TUNER.init(&id, move || {
        let mut autotune =
            Autotune::measured_on(RoutedMatmulTensors::key, "spread-routes", spread_routes);
        let microkernel = autotune.group("microkernel", routines);
        let autotune = autotune.bounded(|_, tensors| tensors.roofline());
        let candidates = RoutedCandidates::new(&hw);
        let autotune = raced(
            autotune,
            &microkernel,
            PlaneShuffleMatmulRoutine,
            candidates.plane_shuffle,
        );
        let autotune = raced(
            autotune,
            &microkernel,
            CubeStagedMatmulRoutine,
            candidates.cube_staged,
        );
        let autotune = raced(
            autotune,
            &microkernel,
            UnitRegistersMatmulRoutine,
            candidates.unit_registers,
        );
        let autotune = floored(autotune, PlaneShuffleMatmulRoutine, floors.plane_shuffle);
        let autotune = floored(autotune, CubeStagedMatmulRoutine, floors.cube_staged);
        floored(autotune, UnitRegistersMatmulRoutine, floors.unit_registers).finish()
    });

    let product = TUNER.execute(&id, &client, tunables, tensors);
    Ok(product.handle)
}

/// The operands a candidate is measured on: the call's rows and stack, each row's slots routed
/// round the experts in turn. The rows are the call's own, not grown to the key's bucket: a
/// candidate is timed on the work the call does, as the m-grouped autotune does.
///
/// The environment build discards the launches that make the call's routes, so they hold
/// whatever the buffer did, and a kernel reading them as indices runs on garbage. Uploaded rather
/// than launched, so a process mode that discards launches keeps them.
pub(super) fn spread_routes(
    _key: &RoutedMatmulKey,
    tensors: &RoutedMatmulTensors,
) -> RoutedMatmulTensors {
    let (rows, slots, experts) = (tensors.rows(), tensors.slots(), tensors.experts());
    let routes: Vec<u8> = (0..rows * slots)
        .flat_map(|route| ((route % experts) as u32).to_le_bytes())
        .collect();
    let client = tensors.client();
    let routes = TensorHandle::new(
        client.create_from_slice(&routes),
        [rows, slots],
        [slots, 1],
        ElemType::UInt(UIntKind::U32),
    );
    tensors.with_routes(PlainTensor::of_handle(routes))
}

type RoutedAutotune = Autotune<RoutedMatmulKey, RoutedMatmulTensors, MatmulProduct>;

/// `autotune` with every setting of `listings` added to `group` as a candidate of `routine`, in
/// their order.
fn raced<Mk, SM, R>(
    autotune: RoutedAutotune,
    group: &TuneGroup<RoutedMatmulKey>,
    routine: R,
    listings: Vec<Listing<R::Settings>>,
) -> RoutedAutotune
where
    Mk: MatmulMicrokernel,
    R: MatmulRoutine<Mk, SM> + Copy + Send + Sync + 'static,
    R::Settings: Send + Sync + 'static,
{
    listings.into_iter().fold(autotune, |autotune, listing| {
        autotune.candidate(group, candidate(routine, listing.settings, listing.kind))
    })
}

/// `autotune` with every setting of `floors` added as a floor of `routine`, in their order.
fn floored<Mk, SM, R>(
    autotune: RoutedAutotune,
    routine: R,
    floors: impl IntoIterator<Item = Listing<R::Settings>>,
) -> RoutedAutotune
where
    Mk: MatmulMicrokernel,
    R: MatmulRoutine<Mk, SM> + Copy + Send + Sync + 'static,
    R::Settings: Send + Sync + 'static,
{
    floors.into_iter().fold(autotune, |autotune, floor| {
        autotune.floor(candidate(routine, floor.settings, floor.kind))
    })
}

/// `routine` at `settings`, launched on the routed product of the stacks it is `listed` for,
/// under the setting's name.
fn candidate<Mk, SM, R>(
    routine: R,
    settings: R::Settings,
    listed: ListedFor,
) -> Tunable<RoutedMatmulKey, RoutedMatmulTensors, MatmulProduct>
where
    Mk: MatmulMicrokernel,
    R: MatmulRoutine<Mk, SM> + Send + Sync + 'static,
    R::Settings: Send + Sync + 'static,
{
    Tunable::new(
        &settings.to_string(),
        move |tensors: RoutedMatmulTensors| {
            listed
                .admits(tensors.product())
                .map_err(|refusal| format!("{settings} {refusal:?}"))?;
            let hw = Hardware::from_client(tensors.client());
            tensors
                .launch(&hw, &routine, &settings)
                .map(|handle| MatmulProduct {
                    client: tensors.client().clone(),
                    handle,
                })
                .map_err(|reason| format!("{settings} does not serve this problem: {reason:?}"))
        },
    )
}
