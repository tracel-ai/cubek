use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_std::routine::Routine;

use crate::kernels::indexed::routed::RoutedMatmulTensors;
use crate::kernels::tune::{ListedFor, Listing, Product, first_served};
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
    pub plane_shuffle: [Listing<PlaneShuffleMatmulRoutineSettings>; 2],
    pub cube_staged: [Listing<CubeStagedMatmulRoutineSettings>; 2],
    pub unit_registers: [Listing<UnitRegistersMatmulRoutineSettings>; 1],
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
        let plane_shuffle = Listing::admitted(&self.plane_shuffle, stack)
            .map(|floor| tensors.served_by(hw, &PlaneShuffleMatmulRoutine, &floor.settings));
        let cube_staged = Listing::admitted(&self.cube_staged, stack)
            .map(|floor| tensors.served_by(hw, &CubeStagedMatmulRoutine, &floor.settings));
        let unit_registers = Listing::admitted(&self.unit_registers, stack)
            .map(|floor| tensors.served_by(hw, &UnitRegistersMatmulRoutine, &floor.settings));
        first_served(plane_shuffle.chain(cube_staged).chain(unit_registers))
    }

    pub(crate) const LISTED: Self = Self {
        plane_shuffle: [
            Listing {
                settings: PlaneShuffleMatmulRoutineSettings::DEFAULT,
                kind: ListedFor::Either,
            },
            Listing {
                settings: PlaneShuffleMatmulRoutineSettings::PACKED,
                kind: ListedFor::Either,
            },
        ],
        cube_staged: [
            Listing {
                settings: CubeStagedMatmulRoutineSettings::DEFAULT,
                kind: ListedFor::Float,
            },
            Listing {
                settings: CubeStagedMatmulRoutineSettings::PACKED,
                kind: ListedFor::Packed,
            },
        ],
        unit_registers: [Listing {
            settings: UnitRegistersMatmulRoutineSettings::DEFAULT,
            kind: ListedFor::Float,
        }],
    };
}

/// The settings worth racing on a routed product, each routine's list for the stacks each is
/// listed for.
#[derive(Debug, Clone)]
pub(crate) struct RoutedCandidates {
    /// Every plane shuffle setting worth racing. Each declines a stack it cannot read, so the
    /// float ones race on a packed stack too: some read one faster than the packed lead does.
    pub plane_shuffle: Vec<Listing<PlaneShuffleMatmulRoutineSettings>>,
    /// Every cube staged setting worth racing: against a packed stack the one-step settings
    /// first, then the packed ones.
    pub cube_staged: Vec<Listing<CubeStagedMatmulRoutineSettings>>,
    /// Every unit registers setting worth racing, listed for a float stack.
    pub unit_registers: Vec<Listing<UnitRegistersMatmulRoutineSettings>>,
}

impl RoutedCandidates {
    /// The settings worth racing on `hw`.
    pub(crate) fn new(hw: &Hardware) -> Self {
        let plane_shuffle = Listing::float_then_packed(
            PlaneShuffleMatmulRoutine.settings(hw),
            PlaneShuffleMatmulRoutine.packed_settings(),
        );
        let mut packed = CubeStagedMatmulRoutine.packed_one_step_settings();
        let rest: Vec<_> = CubeStagedMatmulRoutine
            .packed_settings(hw)
            .into_iter()
            .filter(|settings| !packed.contains(settings))
            .collect();
        packed.extend(rest);
        let cube_staged = Listing::merged(CubeStagedMatmulRoutine.settings(hw), packed);
        let unit_registers = UnitRegistersMatmulRoutine
            .settings(hw)
            .into_iter()
            .map(|settings| Listing {
                settings,
                kind: ListedFor::Float,
            })
            .collect();
        Self {
            plane_shuffle,
            cube_staged,
            unit_registers,
        }
    }
}
