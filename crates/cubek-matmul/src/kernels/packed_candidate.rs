//! A routine at one of the settings it lists for a packed input ([`MatmulPackedCandidate`]), as a
//! caller that measures the routines lists and launches it.

use crate::kernels::plan::{Decoded, MatmulCall};
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::{
    CubeShuffleMatmulRoutineSettings, CubeStagedAsyncCopyMatmulRoutineSettings,
    CubeStagedMatmulRoutineSettings, CubeStagedTmaFillerMatmulRoutineSettings,
    HypercubeLastArrivalMatmulRoutineSettings, ListedFor, MatmulRoutine, MatmulTensors,
    PlaneShuffleMatmulRoutineSettings,
};
use cubecl::std::tensor::TensorHandle;
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_microkernel::matmul::cube::CubeShufflePlaneSettings;
use cubek_std::Hardware;
use cubek_std::routine::Routine;

/// What a packed setting contracts on: the one thing of it a caller ordering packed candidates
/// for a key reads.
pub trait PackedContraction {
    /// Whether it contracts on matrix instructions, whose fragments take several of the product's
    /// rows at once, rather than in its units' registers a row at a time.
    fn on_matrix_instructions(&self) -> bool;
}

impl PackedContraction for PlaneShuffleMatmulRoutineSettings {
    fn on_matrix_instructions(&self) -> bool {
        false
    }
}

impl PackedContraction for CubeShuffleMatmulRoutineSettings {
    fn on_matrix_instructions(&self) -> bool {
        matches!(
            self.microkernel.plane,
            CubeShufflePlaneSettings::Fragment(_)
        )
    }
}

impl PackedContraction for CubeStagedMatmulRoutineSettings {
    fn on_matrix_instructions(&self) -> bool {
        true
    }
}

impl PackedContraction for CubeStagedAsyncCopyMatmulRoutineSettings {
    fn on_matrix_instructions(&self) -> bool {
        true
    }
}

impl PackedContraction for CubeStagedTmaFillerMatmulRoutineSettings {
    fn on_matrix_instructions(&self) -> bool {
        true
    }
}

impl PackedContraction for HypercubeLastArrivalMatmulRoutineSettings {
    fn on_matrix_instructions(&self) -> bool {
        true
    }
}

/// One routine at one of the settings it lists for a packed input, reading the packed inputs as
/// stored, on the device it was listed for. A packed routine's launch reads how a packed input is
/// stored and how its device loads it, which a key does not carry, so no launch is prepared for a
/// key.
#[derive(Debug, Clone)]
pub struct MatmulPackedCandidate<R: Routine> {
    routine: R,
    settings: R::Settings,
    hw: Hardware,
}

impl<R: Routine> MatmulPackedCandidate<R> {
    /// The setting this candidate launches the routine at, which names it.
    pub fn settings(&self) -> &R::Settings {
        &self.settings
    }
}

impl<R: Routine + Copy> MatmulPackedCandidate<R> {
    /// `routine` at each of `settings` on `hw`, in their order.
    pub fn of(
        routine: R,
        settings: impl IntoIterator<Item = R::Settings>,
        hw: &Hardware,
    ) -> Vec<Self> {
        settings
            .into_iter()
            .map(|settings| Self {
                routine,
                settings,
                hw: hw.clone(),
            })
            .collect()
    }

    /// The routine this candidate launches.
    pub fn routine(&self) -> R {
        self.routine
    }

    /// `call`'s product by this candidate's routine at its setting, reading the packed inputs as
    /// stored, or the reason it did not run: into its destination
    /// ([`MatmulRoutine::launch_into`]) or into an output the launch allocates, written in the
    /// call's element.
    ///
    /// # Errors
    ///
    /// Two float inputs, which the packed settings are not listed for, before compiling anything,
    /// what [`MatmulTensors::written_in`] declines, and why the routine cannot run the product at
    /// this setting.
    pub fn launch<M: MatmulMicrokernel, SM>(&self, call: MatmulCall) -> Result<TensorHandle, String>
    where
        R: MatmulRoutine<M, SM>,
    {
        ListedFor::Packed
            .admits(call.product())
            .map_err(|decline| format!("{decline:?}"))?;
        let settings = &self.settings;
        let (lhs, rhs) = (call.lhs.tensor, call.rhs.tensor);
        let launched = match call.destination {
            Some(destination) => self
                .routine
                .launch_into(
                    &self.hw,
                    &call.client,
                    lhs,
                    rhs,
                    PlainTensor::of_handle(destination.clone()),
                    settings,
                )
                .map(|()| destination),
            None => {
                MatmulTensors::written_in(&call.client, lhs, rhs, call.out).and_then(|tensors| {
                    let (blueprint, grid) = R::prepare(&self.hw, &tensors.problem(), settings)?;
                    self.routine
                        .dispatch_into_buffer(&tensors, blueprint, grid)?;
                    Ok(tensors
                        .into_out()
                        .expect("a launch allocating its output hands it back"))
                })
            }
        };
        launched.map_err(|why| format!("{settings} does not serve this problem: {why:?}"))
    }

    /// [`launch`](Self::launch) with a packed lhs decoded first, the rhs still read as stored: a
    /// packed activation beside a weight no routine reads it beside (an MXFP8 activation's blocks
    /// of 32 beside an NVFP4 weight's 16), or beside one a routine reads it beside slower than the
    /// weight alone, contracted as fast as the weight alone allows. An activation is a few rows
    /// against the weight's many, so decoding it costs little of a product that streams the
    /// weight; decoding the weight instead, as a floor decoding every packed input does, rewrites
    /// the weight on every call.
    ///
    /// # Errors
    ///
    /// A float lhs, which the candidate as stored serves; and every error of
    /// [`launch`](Self::launch).
    pub fn launch_activation_decoded<M: MatmulMicrokernel, SM>(
        &self,
        call: MatmulCall,
    ) -> Result<TensorHandle, String>
    where
        R: MatmulRoutine<M, SM>,
    {
        if matches!(call.lhs.tensor, StoredTensor::Plain(_)) {
            return Err("the lhs is float: the candidate reading it as stored serves it".into());
        }
        let lhs = Decoded
            .input(&call, &call.lhs)
            .map_err(|why| why.to_string())?;
        self.launch(MatmulCall { lhs, ..call })
    }
}
