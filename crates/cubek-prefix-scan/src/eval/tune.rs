use crate::{
    base::{
        PrefixScanGlobalStrategy, PrefixScanKind, PrefixScanLocalStrategy, PrefixScanSettings,
        PrefixScanStrategy, strategies,
    },
    instructions::PrefixScanInstructionKind,
    launch::prefix_scan_launch,
};
use cubecl::{
    AutotuneKey, CubeTuneId, Device,
    client::Client,
    prelude::*,
    std::tensor::TensorHandle,
    tune::{LocalTuner, Tunable, local_tuner},
    zspace::Shape,
};
use cubek_std::autotune::{Autotune, CandidatePriority, GroupPriority, TuneCandidate};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, AutotuneKey)]
pub struct PrefixScanTuneKey {
    pub dtype: ElemType,
    pub shape: Shape,
    pub scan_kind: PrefixScanKind,
    pub scan_instruction_kind: PrefixScanInstructionKind,
}

#[derive(Clone)]
pub struct PrefixScanTuneInputs {
    pub input: TensorHandle,
    pub output: TensorHandle,
    pub scan_kind: PrefixScanKind,
    pub scan_instruction_kind: PrefixScanInstructionKind,
}

type PrefixScanAutotune = Autotune<PrefixScanTuneKey, PrefixScanTuneInputs, ()>;

pub struct PrefixScanCandidate {
    pub settings: PrefixScanSettings,
}

impl PrefixScanCandidate {
    pub fn launch(&self, client: &Client, inputs: PrefixScanTuneInputs) -> Result<(), String> {
        let dtype = inputs.input.dtype;
        prefix_scan_launch(
            client,
            inputs.input.binding(),
            inputs.output.binding(),
            dtype,
            inputs.scan_kind,
            inputs.scan_instruction_kind,
            &self.settings,
        )
        .map_err(|error| error.to_string())
    }
}

impl TuneCandidate for PrefixScanCandidate {
    type Strategy = PrefixScanSettings;
    type Key = PrefixScanTuneKey;
    type Blueprint = ();

    fn strategy(&self) -> &Self::Strategy {
        &self.settings
    }

    fn priority(&self, _key: &PrefixScanTuneKey) -> CandidatePriority {
        CandidatePriority::FIRST
    }
}

pub fn prefix_scan_launch_tuned(
    device: &Device,
    inputs: PrefixScanTuneInputs,
    routines: GroupPriority,
) {
    let client = device.client().clone();
    let id = CubeTuneId::new(&client, device);
    static TUNER: LocalTuner<PrefixScanTuneKey, CubeTuneId> = local_tuner!("prefix-scan");
    let set_client = client.clone();
    let tunables = TUNER.init(&id, move || {
        let mut autotune =
            PrefixScanAutotune::new(|inputs: &PrefixScanTuneInputs| PrefixScanTuneKey {
                dtype: inputs.input.dtype,
                shape: inputs.input.shape().clone(),
                scan_kind: inputs.scan_kind,
                scan_instruction_kind: inputs.scan_instruction_kind,
            });
        let mut candidates = Vec::new();
        for strategy in strategies() {
            for vector_size_shift in 0..4 {
                for units_per_cube_shift in 0..4 {
                    candidates.push(PrefixScanCandidate {
                        settings: PrefixScanSettings {
                            vector_size: 1 << vector_size_shift,
                            units_per_cube: set_client.properties().hardware.max_units_per_cube
                                >> units_per_cube_shift,
                            strategy: strategy.clone(),
                        },
                    });
                }
            }
        }
        let prioritized_client = set_client.clone();
        let floor_client = set_client.clone();
        let routines = autotune.group("routine", routines);
        autotune = autotune.prioritized(&routines, candidates, move |candidate, inputs| {
            candidate.launch(&prioritized_client, inputs)
        });
        autotune
            .floor(Tunable::new(
                "prefix-scan-floor",
                move |inputs: PrefixScanTuneInputs| {
                    PrefixScanCandidate {
                        settings: PrefixScanSettings::new(
                            floor_client.properties(),
                            inputs.input.dtype,
                            PrefixScanStrategy {
                                cube: PrefixScanLocalStrategy::KoggeStoneOrHillisSteele,
                                global: PrefixScanGlobalStrategy::StreamScan,
                            },
                        ),
                    }
                    .launch(&floor_client, inputs)
                },
            ))
            .finish()
    });
    TUNER.execute(&id, &client, tunables, inputs);
}
