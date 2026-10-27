//! The routed product's autotune, `out[r, s] = lhs[r] @ stacked[routes[r, s]]`: the dense plane
//! shuffle routine's settings over the dense unit routine at its default, each launched on the
//! product's batches of one row ([`RoutedMatmulTensors`]), keyed by the stack's shape and a row's
//! slots. What no routine serves is the caller's to run another way.

use cubecl::ir::ElemType;
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubecl::tune::{LocalTuner, Tunable, local_tuner};
use cubecl::{self as cubecl, AutotuneKey, CubeTuneId};
use cubecl_common::device::Device;
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_std::Hardware;
use cubek_std::autotune::{Autotune, GroupPriority};
use cubek_std::routine::Routine;
use serde::{Deserialize, Serialize};

use crate::kernels::indexed::routed::RoutedMatmulTensors;
use crate::kernels::indexed::stack::SlabAxis;
use crate::kernels::select::problem::OperandKind;
use crate::kernels::tune::key::OperandLayout;
use crate::kernels::{
    MatmulRoutine, PlaneShuffleMatmulRoutine, PlaneShuffleMatmulRoutineSettings,
    UnitRegistersMatmulRoutine, UnitRegistersMatmulRoutineSettings,
};

/// What makes two routed products want the same launch.
///
/// The contraction, one expert's columns, the slots and the expert count are **exact**: they are
/// the model config's, so the key space is that architecture's handful of shapes. The row count is
/// anchored, being the axis a caller varies freely (batch × sequence) and the one that decides
/// whether the grid is full.
#[derive(Hash, Eq, PartialEq, Debug, Clone, Serialize, Deserialize, AutotuneKey)]
pub struct RoutedMatmulKey {
    /// The lhs and the output bind at this element.
    elem: ElemType,
    /// The contraction.
    k: usize,
    /// One expert's output columns.
    n: usize,
    /// Experts in the stack.
    experts: usize,
    /// Experts each row selected.
    slots: usize,
    /// The lhs's rows, anchored to the power of two above them.
    #[autotune(anchor(exp(min = 1, max = 65536, base = 2)))]
    m: usize,
    /// Whether the stack is float or packed, and how: a packed stack is read packed, by other
    /// kernels than a float one.
    stack: OperandKind,
    /// Which axis the stack lines along. Two stacks of identical shape laid out differently are
    /// different problems: the binding reads opposite dims.
    lines: OperandLayout,
    /// Which axis the slabs lie along: a stack read as stored and the same stack read transposed
    /// put the experts on different dims.
    slabs: SlabAxis,
}

/// What the routed autotune measures a candidate on: the operands, and the output every
/// candidate writes, so the call's output is the one the winner wrote.
#[derive(Clone)]
struct RoutedMatmulInputs {
    tensors: RoutedMatmulTensors,
    out: TensorHandle,
}

impl RoutedMatmulInputs {
    /// The key the autotune measures these inputs under.
    fn key(&self) -> RoutedMatmulKey {
        self.tensors.key()
    }
}

/// The floors' settings, named once because the entry asks them of a problem too: a round must
/// reach a candidate that answers. The unit routine's serves a float stack, the plane shuffle's
/// lead against a packed rhs a packed one.
const UNIT_FLOOR: UnitRegistersMatmulRoutineSettings = UnitRegistersMatmulRoutineSettings::DEFAULT;
const PLANE_SHUFFLE_FLOOR: PlaneShuffleMatmulRoutineSettings =
    PlaneShuffleMatmulRoutineSettings::PACKED;

/// The routed product `tensors` states, through whichever setting measures fastest for it on
/// `device`: `[rows, slots, n]` at the lhs's element. `routines` is the priority of the plane
/// shuffle's group; the floors sit under it.
///
/// # Errors
///
/// A product the floor does not serve, a packed stack among them, so no candidate of the
/// autotune is known to: the caller names what runs instead, since a round with no candidate that
/// answers is a panic inside the tuner.
pub fn launch_tuned(
    device: &impl Device,
    tensors: RoutedMatmulTensors,
    routines: GroupPriority,
) -> Result<TensorHandle, Refusal> {
    let client = tensors.client().clone();
    let hw = Hardware::from_client(&client);
    // No row is an empty answer, which no round measures.
    if tensors.rows() == 0 {
        return tensors.launch(&hw, &UnitRegistersMatmulRoutine, &UNIT_FLOOR);
    }
    // A round must reach a floor that answers. The unit routine serves every float stack; a
    // packed one is read only by the plane shuffle, which declines a device that commits to no
    // plane width.
    if tensors.packed() {
        tensors.served_by(&hw, &PlaneShuffleMatmulRoutine, &PLANE_SHUFFLE_FLOOR)?;
    }

    // Named rather than `module_path!()`, so the cached winners stay this autotune's wherever the
    // code lives; a cached winner is an index into the candidate list.
    static TUNER: LocalTuner<RoutedMatmulKey, CubeTuneId> = local_tuner!("routed");

    // Keyed by the device: the set is built from what this device supports.
    let id = CubeTuneId::new(&client, device);
    let tunables = TUNER.init(&id, move || {
        let mut autotune = Autotune::new(RoutedMatmulInputs::key);
        let microkernel = autotune.group("microkernel", routines);
        // The plane shuffle's settings for a float rhs, then those for a packed one: a stack
        // of either races the one list, each setting declining the rhs it does not read.
        let packed = PlaneShuffleMatmulRoutine.packed_settings();
        let float = PlaneShuffleMatmulRoutine.settings(&hw);
        let fresh = |settings: &PlaneShuffleMatmulRoutineSettings| !float.contains(settings);
        let packed: Vec<_> = packed.into_iter().filter(fresh).collect();
        float
            .into_iter()
            .chain(packed)
            .fold(autotune, |autotune, settings| {
                autotune.candidate(&microkernel, candidate(PlaneShuffleMatmulRoutine, settings))
            })
            // The unit routine on the floor rather than in a group, as the dense few-rows autotune
            // has it: it asks nothing of the hardware, so it serves a float stack on a device that
            // commits to no plane width, where the plane shuffle declines. A packed stack's floor
            // is the plane shuffle's lead against a packed rhs.
            .floor(candidate(UnitRegistersMatmulRoutine, UNIT_FLOOR))
            .floor(candidate(PlaneShuffleMatmulRoutine, PLANE_SHUFFLE_FLOOR))
            .finish()
    });

    let out = tensors.output();
    let inputs = RoutedMatmulInputs {
        tensors,
        out: out.clone(),
    };
    TUNER.execute(&id, &client, tunables, inputs);
    Ok(out)
}

/// `routine` at `settings`, launched on the routed product into its inputs' output, under the
/// setting's name.
fn candidate<Mk, SM, R>(
    routine: R,
    settings: R::Settings,
) -> Tunable<RoutedMatmulKey, RoutedMatmulInputs, ()>
where
    Mk: MatmulMicrokernel,
    R: MatmulRoutine<Mk, SM> + Send + Sync + 'static,
    R::Settings: Send + Sync + 'static,
{
    Tunable::new(&settings.to_string(), move |inputs: RoutedMatmulInputs| {
        let hw = Hardware::from_client(inputs.tensors.client());
        inputs
            .tensors
            .launch_into(&hw, &routine, &settings, &inputs.out)
            .map_err(|reason| format!("{settings} does not serve this problem: {reason:?}"))
    })
}
