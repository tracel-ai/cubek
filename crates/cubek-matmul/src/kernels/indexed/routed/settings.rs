use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_std::routine::Routine;

use crate::kernels::indexed::routed::RoutedMatmulTensors;
use crate::kernels::tune::{Listed, ListedFor, Product, first_served};
use crate::kernels::{
    CubeStagedMatmulRoutine, CubeStagedMatmulRoutineSettings, PlaneShuffleMatmulRoutine,
    PlaneShuffleMatmulRoutineSettings, UnitRegistersMatmulRoutine,
    UnitRegistersMatmulRoutineSettings,
};

/// The settings every round must be able to fall back on, each for the kind of stack it serves.
/// Each declines where another runs: the plane shuffle where the device commits to no plane
/// width, its default against a packed stack, which its lead against a packed rhs reads, the cube
/// staged routine on a CPU runtime, which holds fewer units a cube than its planes, and the unit
/// registers, which ask nothing of the device, against a float stack.
///
/// A packed stack on a CPU runtime has no floor: the plane shuffle has no plane width there, the
/// cube staged routine holds too few units a cube, and the unit registers refuse a packed stack.
/// It is refused before racing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RoutedFloors {
    pub plane_shuffle: [Listed<PlaneShuffleMatmulRoutineSettings>; 2],
    pub cube_staged: [Listed<CubeStagedMatmulRoutineSettings>; 2],
    pub unit_registers: [Listed<UnitRegistersMatmulRoutineSettings>; 1],
}

impl RoutedFloors {
    /// Whether some floor listed for a stack of kind `stack` prepares `tensors` on `hw`.
    ///
    /// # Errors
    ///
    /// The last floor's refusal, where none serves. The first that serves ends the walk.
    pub(crate) fn served(
        &self,
        hw: &Hardware,
        stack: Product,
        tensors: &RoutedMatmulTensors,
    ) -> Result<(), Refusal> {
        let plane_shuffle = Listed::admitted(&self.plane_shuffle, stack)
            .map(|floor| tensors.served_by(hw, &PlaneShuffleMatmulRoutine, &floor.settings));
        let cube_staged = Listed::admitted(&self.cube_staged, stack)
            .map(|floor| tensors.served_by(hw, &CubeStagedMatmulRoutine, &floor.settings));
        let unit_registers = Listed::admitted(&self.unit_registers, stack)
            .map(|floor| tensors.served_by(hw, &UnitRegistersMatmulRoutine, &floor.settings));
        first_served(plane_shuffle.chain(cube_staged).chain(unit_registers))
    }

    pub(crate) const LISTED: Self = Self {
        plane_shuffle: [
            Listed {
                settings: PlaneShuffleMatmulRoutineSettings::DEFAULT,
                listed: ListedFor::Either,
            },
            Listed {
                settings: PlaneShuffleMatmulRoutineSettings::PACKED,
                listed: ListedFor::Either,
            },
        ],
        cube_staged: [
            Listed {
                settings: CubeStagedMatmulRoutineSettings::DEFAULT,
                listed: ListedFor::Float,
            },
            Listed {
                settings: CubeStagedMatmulRoutineSettings::PACKED,
                listed: ListedFor::Packed,
            },
        ],
        unit_registers: [Listed {
            settings: UnitRegistersMatmulRoutineSettings::DEFAULT,
            listed: ListedFor::Float,
        }],
    };
}

/// The settings worth racing on a routed product, each routine's list for the stacks each is
/// listed for.
pub(crate) struct RoutedCandidates;

impl RoutedCandidates {
    /// Every plane shuffle setting worth racing on `hw`. Each declines a stack it cannot read, so
    /// the float ones race on a packed stack too: some read one faster than the packed lead does.
    pub(crate) fn plane_shuffle(hw: &Hardware) -> Vec<Listed<PlaneShuffleMatmulRoutineSettings>> {
        let float = PlaneShuffleMatmulRoutine.settings(hw);
        let packed: Vec<_> = PlaneShuffleMatmulRoutine
            .packed_settings()
            .into_iter()
            .filter(|settings| !float.contains(settings))
            .map(|settings| Listed {
                settings,
                listed: ListedFor::Packed,
            })
            .collect();
        float
            .into_iter()
            .map(|settings| Listed {
                settings,
                listed: ListedFor::Either,
            })
            .chain(packed)
            .collect()
    }

    /// Every cube staged setting worth racing on `hw`: against a packed stack the one-step
    /// settings first, then the packed ones.
    pub(crate) fn cube_staged(hw: &Hardware) -> Vec<Listed<CubeStagedMatmulRoutineSettings>> {
        let mut packed = CubeStagedMatmulRoutine.packed_one_step_settings();
        let rest = CubeStagedMatmulRoutine.packed_settings(hw);
        for settings in rest {
            if !packed.contains(&settings) {
                packed.push(settings);
            }
        }
        Listed::merged(CubeStagedMatmulRoutine.settings(hw), packed)
    }

    /// Every unit registers setting worth racing on `hw`, listed for a float stack.
    pub(crate) fn unit_registers(hw: &Hardware) -> Vec<Listed<UnitRegistersMatmulRoutineSettings>> {
        UnitRegistersMatmulRoutine
            .settings(hw)
            .into_iter()
            .map(|settings| Listed {
                settings,
                listed: ListedFor::Float,
            })
            .collect()
    }
}
