//! The candidate a routine is listed as at the settings it lists for a packed input.

use crate::kernels::tensor::{PlainTensor, StoredTensor};
use crate::kernels::tune::key::MatmulTuneKey;
use crate::kernels::{
    CubeShuffleMatmulRoutineSettings, CubeStagedMatmulRoutineSettings, MatmulRoutine,
    PlaneShuffleMatmulRoutineSettings,
};
use cubecl::client::Client;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::MatmulMicrokernel;
use cubek_microkernel::matmul::cube::CubeShufflePlaneSettings;
use cubek_std::Hardware;
use cubek_std::autotune::{CandidatePriority, TuneCandidate};
use cubek_std::routine::Routine;

/// What a [`MatmulPackedCandidate`] declines: two float inputs, which its settings were not listed
/// for.
pub const PACKED_SETTING_FLOAT_INPUTS: Refusal = Refusal::Declined {
    why: "a setting listed for a packed input reads one, and both inputs are float",
};

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

    /// `lhs · rhs` on `client`'s device written into `out`, which the caller allocated, by this
    /// candidate's routine at its setting ([`MatmulRoutine::launch_into`]).
    ///
    /// # Errors
    ///
    /// [`PACKED_SETTING_FLOAT_INPUTS`] where neither input is packed, before compiling anything,
    /// and as [`MatmulRoutine::launch_into`].
    pub fn launch_into<M: MatmulMicrokernel, SM>(
        &self,
        client: &Client,
        lhs: StoredTensor,
        rhs: StoredTensor,
        out: PlainTensor,
    ) -> Result<(), Refusal>
    where
        R: MatmulRoutine<M, SM>,
    {
        if let (StoredTensor::Plain(_), StoredTensor::Plain(_)) = (&lhs, &rhs) {
            return Err(PACKED_SETTING_FLOAT_INPUTS);
        }
        self.routine
            .launch_into(&self.hw, client, lhs, rhs, out, &self.settings)
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
