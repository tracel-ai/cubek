//! A routine of the SwiGLU family at one of its settings ([`SwigluRoutineCandidate`]), as a caller
//! lists, launches and identifies it, and as a caller that measures the routines races it.

use crate::kernels::{
    CubeSwigluRoutine, CubeSwigluRoutineSettings, SwigluBinding, SwigluBlueprint, SwigluCall,
    SwigluKey, SwigluRoutine, SwigluRoutinePrepared, SwigluRoutineProblem, SwigluTensors,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::kind::Write;
use cubecl::tile::launch::{Buffered, Decline, DestinationLaunch};
use cubek_matmul::kernels::tensor::{LhsOutput, PackedTensor, PlainTensor};
use cubek_microkernel::row_store::Encoded;
use cubek_std::Hardware;
use cubek_std::routine::{Grid, PreparedLaunches, Routine};
use std::sync::Arc;

/// A routine of this kind at one of its settings, prepared for the device it was listed for, and
/// the launches of the gate written once: each problem prepared once by every clone.
#[derive(Debug, Clone)]
pub struct SwigluRoutineCandidate<R: Routine> {
    routine: R,
    settings: R::Settings,
    hw: Hardware,
    /// The launches this candidate prepared, shared by its clones: they launch the same routine
    /// at the same setting on the same device.
    prepared: Arc<PreparedLaunches<R>>,
}

impl<R: Routine> SwigluRoutineCandidate<R> {
    /// `routine` at `settings`, on `hw`, having prepared nothing yet.
    pub fn new(routine: R, settings: R::Settings, hw: &Hardware) -> Self {
        Self {
            routine,
            settings,
            hw: hw.clone(),
            prepared: Arc::default(),
        }
    }

    /// The setting this candidate launches the routine at.
    pub fn settings(&self) -> &R::Settings {
        &self.settings
    }
}

impl SwigluRoutineCandidate<CubeSwigluRoutine> {
    /// The cube routine at its default, which gates every float row on every device: the floor of
    /// a caller that measures the routines.
    pub fn floor(hw: &Hardware) -> Self {
        Self::new(CubeSwigluRoutine, CubeSwigluRoutineSettings::DEFAULT, hw)
    }
}

impl<R: Routine + Copy> SwigluRoutineCandidate<R> {
    /// Every setting `routine` lists on `hw`, as candidates, its default first.
    pub fn list(routine: R, hw: &Hardware) -> Vec<Self> {
        R::settings(hw)
            .into_iter()
            .map(|settings| Self::new(routine, settings, hw))
            .collect()
    }
}

impl<R, M> SwigluRoutineCandidate<R>
where
    R: SwigluRoutine<M> + Routine<Microkernel = M, Blueprint = SwigluBlueprint<M>>,
{
    /// `silu(gate) ⊙ up` over `both`, `(..., 2 · d)` laid out `[gate | up]`, in `f32`, launched
    /// on its device ([`SwigluTensors::new`]).
    ///
    /// # Errors
    ///
    /// Why this device cannot run the gate at this setting.
    pub fn launch(&self, call: &SwigluCall) -> Result<TensorHandle, Decline> {
        self.launch_rows(SwigluTensors::new(&call.client, call.both.clone())?)
    }

    /// The gate of [`launch`](Self::launch), written into `out`, a contiguous buffer the caller
    /// allocated ([`SwigluTensors::allocate_out`]), and handed back: what every candidate of a race
    /// writes into.
    ///
    /// # Errors
    ///
    /// Why this device cannot run the gate at this setting.
    ///
    /// # Panics
    ///
    /// On an `out` of another shape or element than the gate's.
    pub fn launch_contiguous(
        &self,
        call: &SwigluCall,
        out: TensorHandle,
    ) -> Result<TensorHandle, Decline> {
        self.launch_rows(SwigluTensors::contiguous(
            &call.client,
            call.both.clone(),
            out,
        )?)
    }

    /// The gate of [`launch`](Self::launch), written into `out`, a float `[rows, d]` matrix
    /// stored in tiles the caller allocated, the gated rows landing in its tiles as they are
    /// written ([`SwigluTensors::tiled`]).
    ///
    /// # Errors
    ///
    /// Why this device cannot run the gate at this setting, or why `out` cannot hold it.
    pub fn launch_tiled(&self, call: &SwigluCall, out: PlainTensor) -> Result<(), Decline> {
        let tensors = SwigluTensors::tiled(&call.client, call.both.clone(), out)?;
        self.launch_on::<Buffered, _>(&tensors, |blueprint, grid| {
            tensors
                .bind(blueprint, grid)
                .map(|binding| binding.map_out(|out| (out, Write::Replace)))
        })
    }

    /// The gate of [`launch`](Self::launch), encoded into `out`, a packed `[rows, d]` matrix the
    /// caller allocated under one level of block scales along each row: each block's scale
    /// calibrated from the gated rows rounded to their
    /// element, so the result is the gate's output quantized ([`SwigluTensors::packed`]).
    ///
    /// # Errors
    ///
    /// Why this device cannot encode the gate at this setting, or why `out` cannot hold it.
    pub fn launch_packed(&self, call: &SwigluCall, out: PackedTensor) -> Result<(), Decline> {
        let tensors = SwigluTensors::packed(&call.client, call.both.clone(), out)?;
        self.launch_on::<Encoded, _>(&tensors, |blueprint, grid| tensors.bind(blueprint, grid))
    }

    /// The gate of [`launch`](Self::launch), written into `out` as a product reads its lhs:
    /// [`launch_packed`](Self::launch_packed) or [`launch_tiled`](Self::launch_tiled).
    ///
    /// # Errors
    ///
    /// Those of the launch `out` names.
    pub fn launch_into(&self, call: &SwigluCall, out: LhsOutput) -> Result<(), Decline> {
        match out {
            LhsOutput::Packed(out) => self.launch_packed(call, out),
            LhsOutput::Tiled(out) => self.launch_tiled(call, out),
        }
    }

    /// The launch this candidate prepares for the problem `key` describes, or `None` where the
    /// routine declines it on the device it was listed for. Not held: a key's problem places the
    /// candidate in a round, and may never be launched.
    pub fn prepared(&self, key: &SwigluKey) -> Option<SwigluRoutinePrepared<M>> {
        R::prepare(&self.hw, &SwigluRoutineProblem::from(key), &self.settings)
            .ok()
            .map(|(blueprint, grid)| SwigluRoutinePrepared::new(blueprint, grid))
    }
}

impl<R, M> SwigluRoutineCandidate<R>
where
    R: SwigluRoutine<M> + Routine<Microkernel = M, Blueprint = SwigluBlueprint<M>>,
{
    /// The gate on `tensors`, bound by `bind` and written through `O`. No elements is no launch: a
    /// grid of no cubes is one no device spawns.
    fn launch_on<O: DestinationLaunch, Out>(
        &self,
        tensors: &SwigluTensors<Out>,
        bind: impl FnOnce(&SwigluBlueprint<M>, &Grid) -> Result<SwigluBinding<O::Operand>, Decline>,
    ) -> Result<(), Decline> {
        if tensors.problem().is_empty() {
            return Ok(());
        }
        let (blueprint, grid) = self.prepare(tensors.problem())?;
        let binding = bind(&blueprint, &grid)?;
        self.routine
            .dispatch::<O, _>(tensors, binding, blueprint, grid);
        Ok(())
    }

    /// The launch this candidate prepared for `problem`, or prepares now and holds.
    fn prepare(
        &self,
        problem: &SwigluRoutineProblem,
    ) -> Result<(SwigluBlueprint<M>, Grid), Decline> {
        self.prepared.prepare(&self.hw, problem, &self.settings)
    }

    /// The gate written into the rows `tensors` hold, handed back.
    fn launch_rows(&self, tensors: SwigluTensors) -> Result<TensorHandle, Decline> {
        self.launch_on::<Buffered, _>(&tensors, |blueprint, grid| {
            tensors
                .bind(blueprint, grid)
                .map(|binding| binding.map_out(|out| (out, Write::Replace)))
        })?;
        Ok(tensors.into_out())
    }
}

/// The candidate's name: its setting's, which is what a persisted winner is checksummed by.
impl<R: Routine<Settings: core::fmt::Display>> core::fmt::Display for SwigluRoutineCandidate<R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.settings.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// On devices a dev machine cannot show, every setting prepares: the routine is the floor, so a
    /// decline would leave a key with nothing to measure.
    #[test]
    fn every_device_prepares_every_setting() {
        for key in [
            SwigluKey::new(F16, 8192, 1),
            SwigluKey::new(F16, 2816, 512 * 4),
            SwigluKey::new(F16, 111, 3),
        ] {
            for hw in [
                devices::rdna_range(),
                devices::cdna_wave64(),
                devices::h100(),
                devices::wgpu_no_subgroups(),
                devices::cpu_12(),
            ] {
                for candidate in SwigluRoutineCandidate::list(CubeSwigluRoutine, &hw) {
                    assert!(candidate.prepared(&key).is_some(), "{candidate}");
                }
            }
        }
    }
}
