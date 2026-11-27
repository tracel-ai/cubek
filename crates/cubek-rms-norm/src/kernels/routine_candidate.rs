//! A routine of the RMS-norm family at one of its settings ([`RmsNormRoutineCandidate`]), as a
//! caller lists, launches and identifies it, and as a caller that measures the routines races it.

use crate::kernels::{
    CubeGroupsRmsNormRoutine, CubeGroupsRmsNormRoutineSettings, RMS_NORM_SINK_TAKES_NO_ROWS,
    RmsNormBinding, RmsNormBlueprint, RmsNormCall, RmsNormKey, RmsNormRoutine,
    RmsNormRoutinePrepared, RmsNormRoutineProblem, RmsNormRoutineSinkLaunch, RmsNormTensors,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::kind::Write;
use cubecl::tile::launch::{Buffered, Decline, DestinationLaunch};
use cubek_matmul::kernels::tensor::{LhsOutput, PlainTensor, StoredTensor};
use cubek_microkernel::row_store::Encoded;
use cubek_std::Hardware;
use cubek_std::routine::{Grid, PreparedLaunches, Routine};
use std::sync::Arc;

/// A routine of this kind at one of its settings, prepared for the device it was listed for, and
/// the launches of the norm written once: each problem prepared once by every clone.
#[derive(Debug, Clone)]
pub struct RmsNormRoutineCandidate<R: Routine> {
    routine: R,
    settings: R::Settings,
    hw: Hardware,
    /// The launches this candidate prepared, shared by its clones: they launch the same routine
    /// at the same setting on the same device.
    prepared: Arc<PreparedLaunches<R>>,
}

impl<R: Routine> RmsNormRoutineCandidate<R> {
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

impl RmsNormRoutineCandidate<CubeGroupsRmsNormRoutine> {
    /// The cube groups at their default, which serve every float row on every device: the floor of
    /// a caller that measures the routines.
    pub fn floor(hw: &Hardware) -> Self {
        Self::new(
            CubeGroupsRmsNormRoutine,
            CubeGroupsRmsNormRoutineSettings::DEFAULT,
            hw,
        )
    }
}

impl<R: Routine + Copy> RmsNormRoutineCandidate<R> {
    /// Every setting `routine` lists on `hw`, as candidates, its default first.
    pub fn list(routine: R, hw: &Hardware) -> Vec<Self> {
        R::settings(hw)
            .into_iter()
            .map(|settings| Self::new(routine, settings, hw))
            .collect()
    }
}

impl<R, M> RmsNormRoutineCandidate<R>
where
    R: RmsNormRoutine<M> + Routine<Microkernel = M, Blueprint = RmsNormBlueprint<M>>,
{
    /// `input / sqrt(mean(input², last) + epsilon) · weight`, the square-sum in `f32`, launched on
    /// the operands' device ([`RmsNormTensors::new`]).
    ///
    /// # Errors
    ///
    /// Why this device cannot run the norm at this setting.
    pub fn launch(&self, call: &RmsNormCall) -> Result<TensorHandle, Decline> {
        let tensors = RmsNormTensors::new(&call.client, call.input.clone(), call.weight.clone())?;
        self.launch_rows(call, tensors)
    }

    /// The norm of [`launch`](Self::launch), written into `out`, a contiguous buffer in the
    /// input's shape and element that the caller allocated, and handed back: what every candidate
    /// of a race writes into.
    ///
    /// # Errors
    ///
    /// Why this device cannot run the norm at this setting, and
    /// [`RMS_NORM_OUTPUT_UNREAD`](crate::kernels::RMS_NORM_OUTPUT_UNREAD) for an `out` of another
    /// shape or element than the input's, or not dense.
    pub fn launch_contiguous(
        &self,
        call: &RmsNormCall,
        out: TensorHandle,
    ) -> Result<TensorHandle, Decline> {
        let tensors =
            RmsNormTensors::contiguous(&call.client, call.input.clone(), call.weight.clone(), out)?;
        self.launch_rows(call, tensors)
    }

    /// The norm of [`launch`](Self::launch), written into `out`, a float `[rows, dim]` matrix
    /// stored in tiles the caller allocated, its rows landing in its tiles as they are written
    /// ([`RmsNormTensors::tiled`]).
    ///
    /// # Errors
    ///
    /// Why this device cannot run the norm at this setting, or why `out` cannot hold it.
    pub fn launch_tiled(&self, call: &RmsNormCall, out: PlainTensor) -> Result<(), Decline> {
        let tensors =
            RmsNormTensors::tiled(&call.client, call.input.clone(), call.weight.clone(), out)?;
        self.launch_on::<Buffered, _>(call, &tensors, |blueprint, grid| {
            tensors
                .bind(blueprint, grid)
                .map(|binding| binding.map_out(|out| (out, Write::Replace)))
        })
    }

    /// The norm of [`launch`](Self::launch), encoded into `out`, a packed `[rows, dim]` matrix
    /// the caller allocated under one level of block scales along each row: each block's scale
    /// calibrated from the rows rounded to their
    /// element, so the result is the norm's output quantized ([`RmsNormTensors::packed`]).
    ///
    /// # Errors
    ///
    /// Why this device cannot encode the norm at this setting, or why `out` cannot hold it.
    pub fn launch_packed(&self, call: &RmsNormCall, out: StoredTensor) -> Result<(), Decline> {
        let tensors =
            RmsNormTensors::packed(&call.client, call.input.clone(), call.weight.clone(), out)?;
        self.launch_on::<Encoded, _>(call, &tensors, |blueprint, grid| {
            tensors.bind(blueprint, grid)
        })
    }

    /// The norm of [`launch`](Self::launch), written into `out` as a product reads its lhs:
    /// [`launch_packed`](Self::launch_packed) or [`launch_tiled`](Self::launch_tiled).
    ///
    /// # Errors
    ///
    /// Those of the launch `out` names.
    pub fn launch_into(&self, call: &RmsNormCall, out: LhsOutput) -> Result<(), Decline> {
        match out {
            LhsOutput::Packed(out) => self.launch_packed(call, out),
            LhsOutput::Tiled(out) => self.launch_tiled(call, out),
        }
    }

    /// The norm of `input`, scaled by `weight`, handed to a sink.
    ///
    /// # Errors
    ///
    /// [`RMS_NORM_SINK_TAKES_NO_ROWS`]; what [`RmsNormTensors::sink`] declines; why this device
    /// cannot run the norm at this setting.
    pub fn sink_launch(&self, call: &RmsNormCall) -> Result<RmsNormRoutineSinkLaunch<R, M>, Decline>
    where
        R: Copy,
    {
        let tensors = RmsNormTensors::sink(&call.client, call.input.clone(), call.weight.clone())?;
        if tensors.problem().rows == 0 {
            return Err(RMS_NORM_SINK_TAKES_NO_ROWS);
        }
        let prepared = self.prepare(tensors.problem())?;
        RmsNormRoutineSinkLaunch::new(self.routine, tensors, prepared, call.epsilon)
    }

    /// The launch this candidate prepares for the problem `key` describes, or `None` where the
    /// routine declines it on the device it was listed for. Not held: a key's problem places the
    /// candidate in a round, and may never be launched.
    pub fn prepared(&self, key: &RmsNormKey) -> Option<RmsNormRoutinePrepared<M>> {
        R::prepare(&self.hw, &RmsNormRoutineProblem::from(key), &self.settings)
            .ok()
            .map(|(blueprint, grid)| RmsNormRoutinePrepared::new(blueprint, grid))
    }
}

impl<R, M> RmsNormRoutineCandidate<R>
where
    R: RmsNormRoutine<M> + Routine<Microkernel = M, Blueprint = RmsNormBlueprint<M>>,
{
    /// The norm of `call` on `tensors`, bound by `bind` and written through `O`. No rows is no
    /// launch: a grid of no cubes is one no device spawns.
    fn launch_on<O: DestinationLaunch, Out>(
        &self,
        call: &RmsNormCall,
        tensors: &RmsNormTensors<Out>,
        bind: impl FnOnce(&RmsNormBlueprint<M>, &Grid) -> Result<RmsNormBinding<O::Operand>, Decline>,
    ) -> Result<(), Decline> {
        if tensors.problem().rows == 0 {
            return Ok(());
        }
        let (blueprint, grid) = self.prepare(tensors.problem())?;
        let binding = bind(&blueprint, &grid)?;
        self.routine
            .dispatch::<O, _>(tensors, binding, blueprint, grid, call.epsilon);
        Ok(())
    }

    /// The launch this candidate prepared for `problem`, or prepares now and holds.
    fn prepare(
        &self,
        problem: &RmsNormRoutineProblem,
    ) -> Result<(RmsNormBlueprint<M>, Grid), Decline> {
        self.prepared.prepare(&self.hw, problem, &self.settings)
    }

    /// The norm of `call` written into the rows `tensors` hold, handed back.
    fn launch_rows(
        &self,
        call: &RmsNormCall,
        tensors: RmsNormTensors,
    ) -> Result<TensorHandle, Decline> {
        self.launch_on::<Buffered, _>(call, &tensors, |blueprint, grid| {
            tensors
                .bind(blueprint, grid)
                .map(|binding| binding.map_out(|out| (out, Write::Replace)))
        })?;
        Ok(tensors.into_out())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::CubeRmsNormRoutine;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// On devices a dev machine cannot show: the cube groups, this kind's floor, prepare
    /// everywhere.
    #[test]
    fn every_device_keeps_the_floor() {
        for key in [
            RmsNormKey::new(F16, 2560, 1),
            RmsNormKey::new(F16, 128, 512 * 24),
            RmsNormKey::new(F16, 8192, 1),
        ] {
            for hw in [
                devices::rdna_range(),
                devices::cdna_wave64(),
                devices::h100(),
                devices::wgpu_no_subgroups(),
                devices::cpu_12(),
            ] {
                for floor in RmsNormRoutineCandidate::list(CubeGroupsRmsNormRoutine, &hw) {
                    assert!(floor.prepared(&key).is_some(), "{}", floor.settings());
                }
            }
        }
    }

    /// Settings that snap to one launch prepare one launch, so a round measures it once: a
    /// 512-wide row leaves the cube's plane counts fewer distinct counts than the list holds, and a
    /// 65536-wide row grows every group to the same units, leaving only the groups a cube apart.
    #[test]
    fn settings_snapping_to_one_launch_prepare_one_launch() {
        fn distinct<P: PartialEq>(prepared: impl Iterator<Item = Option<P>>) -> (usize, usize) {
            let mut seen: Vec<P> = Vec::new();
            let mut listed = 0;
            for launch in prepared {
                listed += 1;
                let launch = launch.expect("the routine prepares the row");
                if !seen.contains(&launch) {
                    seen.push(launch);
                }
            }
            (seen.len(), listed)
        }
        let hw = devices::h100();

        let key = RmsNormKey::new(F16, 512, 512);
        let (cubes, listed) = distinct(
            RmsNormRoutineCandidate::list(CubeRmsNormRoutine, &hw)
                .iter()
                .map(|candidate| candidate.prepared(&key)),
        );
        assert!(cubes < listed, "{cubes} launches of {listed} cube settings");

        let key = RmsNormKey::new(F16, 65536, 1);
        let (groups, listed) = distinct(
            RmsNormRoutineCandidate::list(CubeGroupsRmsNormRoutine, &hw)
                .iter()
                .map(|candidate| candidate.prepared(&key)),
        );
        assert!(
            groups < listed,
            "{groups} launches of {listed} cube-groups settings"
        );
    }
}
