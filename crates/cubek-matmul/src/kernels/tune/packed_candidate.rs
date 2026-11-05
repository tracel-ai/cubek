//! The candidate a routine is listed as at the settings it lists for a packed input.

use crate::kernels::plan::{Decoded, MatmulCall};
use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::tune::key::MatmulTuneKey;
use crate::kernels::tune::{MatmulProduct, RoutineIdentity};
use crate::kernels::{
    CubeShuffleMatmulRoutineSettings, CubeStagedMatmulRoutineSettings,
    HypercubeLastArrivalMatmulRoutineSettings, MatmulRoutine, MatmulTensors,
    PlaneShuffleMatmulRoutineSettings,
};
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_microkernel::matmul::cube::CubeShufflePlaneSettings;
use cubek_std::Hardware;
use cubek_std::autotune::{CandidateIdentity, CandidatePriority, TuneCandidate};
use cubek_std::routine::Routine;

/// What a packed setting contracts on, the one thing of it a packed key's priority reads.
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

impl PackedContraction for HypercubeLastArrivalMatmulRoutineSettings {
    fn on_matrix_instructions(&self) -> bool {
        true
    }
}

/// One routine at one of the settings it lists for a packed input, reading the packed inputs as
/// stored.
///
/// Its priority says little of the problem: a packed routine's plan reads how a packed input is
/// stored and how its device loads it, which a key does not carry, so a packed key tries the
/// settings in the order the routine lists them, and a float key last, where the routine declines
/// it before compiling anything. What a key does say is its rows: at one, a setting on matrix
/// instructions fills a single row of its fragments, the arithmetic of several rows for the bytes a
/// setting in registers reads once, so it is tried after those rather than beside them.
#[derive(Debug, Clone)]
pub struct MatmulPackedCandidate<R: Routine> {
    routine: R,
    settings: R::Settings,
    hw: Hardware,
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

    /// The setting this candidate launches the routine at.
    pub fn settings(&self) -> &R::Settings {
        &self.settings
    }

    /// The routine this candidate launches.
    pub fn routine(&self) -> R {
        self.routine
    }

    /// `call`'s product by this candidate's routine at its setting, reading the packed inputs as
    /// stored, or the reason it did not run: through the call's routes where it states them,
    /// otherwise into its destination ([`MatmulRoutine::launch_into`]) or into an output the
    /// launch allocates, written in the call's element.
    ///
    /// # Errors
    ///
    /// Two float inputs, which the packed settings are not listed for, before compiling anything,
    /// what [`MatmulTensors::written_in`] declines, and why the routine cannot run the product at
    /// this setting.
    pub fn launch<M: MatmulMicrokernel, SM>(
        &self,
        call: MatmulCall,
    ) -> Result<MatmulProduct, String>
    where
        R: MatmulRoutine<M, SM>,
    {
        if !call.is_packed() {
            return Err("listed for a packed input; both are float".into());
        }
        let settings = &self.settings;
        let (lhs, rhs) = (call.lhs.tensor, call.rhs.tensor);
        let launched = match (&call.routes, call.destination) {
            (Some(routes), _) => routes.launch(&call.hw, &self.routine, settings),
            (None, Some(destination)) => self
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
            (None, None) => {
                MatmulTensors::written_in(&call.client, lhs, rhs, call.out).and_then(|tensors| {
                    let (blueprint, grid) =
                        self.routine
                            .prepare(&self.hw, &tensors.problem(), settings)?;
                    self.routine.dispatch_into_buffer(&tensors, blueprint, grid);
                    Ok(tensors
                        .into_out()
                        .expect("a launch allocating its output hands it back"))
                })
            }
        };
        launched
            .map(|handle| MatmulProduct {
                client: call.client,
                handle,
            })
            .map_err(|why| format!("{settings} does not serve this problem: {why:?}"))
    }

    /// [`launch`](Self::launch) with a packed lhs decoded first, the rhs still read as stored: a
    /// packed activation beside a weight no routine reads it beside (an MXFP8 activation's blocks
    /// of 32 beside an NVFP4 weight's 16), or beside one a routine reads it beside slower than the
    /// weight alone, contracted as fast as the weight alone allows. An activation is a few rows
    /// against the weight's many, so decoding it costs little of a product that streams the
    /// weight; decoding the weight instead, as the autotunes' floor does, rewrites the weight on
    /// every call.
    ///
    /// # Errors
    ///
    /// A float lhs, which the candidate as stored serves; and every error of
    /// [`launch`](Self::launch).
    pub fn launch_activation_decoded<M: MatmulMicrokernel, SM>(
        &self,
        call: MatmulCall,
    ) -> Result<MatmulProduct, String>
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

    /// What a [`launch_activation_decoded`](Self::launch_activation_decoded) candidate is
    /// identified by: its setting, named apart from the same setting reading the lhs as stored,
    /// which a caller may launch again as a routine; this one it may not, since its lhs is decoded
    /// first.
    pub fn activation_decoded_identity(&self) -> RoutineIdentity
    where
        R::Settings: core::fmt::Display,
    {
        CandidateIdentity::Named(format!("{} activation decoded", self.settings))
    }
}

impl<R: Routine> TuneCandidate for MatmulPackedCandidate<R>
where
    R::Settings: core::fmt::Display + PackedContraction,
{
    type Strategy = R::Settings;
    type Key = MatmulTuneKey;
    type Blueprint = ();

    fn strategy(&self) -> &R::Settings {
        &self.settings
    }

    fn priority(&self, key: &MatmulTuneKey) -> CandidatePriority {
        let one_row_on_fragments = key.m == 1 && self.settings.on_matrix_instructions();
        match key.is_packed() && !one_row_on_fragments {
            true => CandidatePriority::FIRST,
            false => CandidatePriority::LAST,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::CubeShuffleMatmulRoutine;
    use crate::kernels::select::problem::OperandKind;
    use crate::kernels::tune::key::{MatmulBatching, OperandKey};
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::devices;
    use cubek_std::matmul_axes::K;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// At one row a packed key tries the settings in registers first and those on matrix
    /// instructions after them; past one row it tries every packed setting first. A float key
    /// tries them last.
    #[test]
    fn one_row_tries_the_registers_before_the_matrix_instructions() {
        use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
        let scheme = QuantScheme::default()
            .with_value(QuantValue::E2M1)
            .per_block([16, 1], ScaleDtype::F16)
            .with_store(QuantStore::PackedU32(1));
        let key = |rows, rhs: OperandKind| {
            MatmulTuneKey::new(
                F16,
                5120,
                5120,
                rows,
                1,
                OperandKey::float(Some(K), MatmulBatching::Shared),
                OperandKey {
                    kind: rhs,
                    ..OperandKey::float(Some(K), MatmulBatching::Shared)
                },
            )
        };
        let packed = OperandKind::Quantized(scheme);
        // The instruction a plane decodes a packed rhs into: f16 summed in f32, sixteen by eight.
        let hw = devices::offering_mma(
            devices::h100(),
            [cubecl::ir::features::MmaConfig {
                a_type: F16,
                b_type: F16,
                cd_type: ElemType::Float(FloatKind::F32),
                m: 16,
                n: 8,
                k: 16,
            }],
        );
        let settings = CubeShuffleMatmulRoutine.packed_settings(&hw);
        assert!(
            settings
                .iter()
                .any(PackedContraction::on_matrix_instructions),
            "the device lists settings on matrix instructions"
        );
        for candidate in MatmulPackedCandidate::of(CubeShuffleMatmulRoutine, settings, &hw) {
            let one_row = match candidate.strategy().on_matrix_instructions() {
                true => CandidatePriority::LAST,
                false => CandidatePriority::FIRST,
            };
            let name = candidate.strategy();
            assert_eq!(candidate.priority(&key(1, packed)), one_row, "{name}");
            assert_eq!(
                candidate.priority(&key(8, packed)),
                CandidatePriority::FIRST,
                "{name}"
            );
            assert_eq!(
                candidate.priority(&key(8, OperandKind::Float)),
                CandidatePriority::LAST,
                "{name}"
            );
        }
    }
}
