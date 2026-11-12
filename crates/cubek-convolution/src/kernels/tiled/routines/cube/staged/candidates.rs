use crate::kernels::tiled::{ConvSettings, InstructionShape};
use core::num::NonZeroUsize;
use cubecl::ir::{ElemType, FloatKind};
use cubek_microkernel::matmul::cube::{CubeStagedMicrokernelSettings, MatmulPlaneLayout};
use cubek_microkernel::matmul::plane::{PlaneFragmentMicrokernelSettings, PlaneFragmentTransport};
use cubek_std::{Hardware, Instructions, UNIT_REGISTERS, manual_mma};

/// The share of a unit's registers the accumulators take, the rest left to the operand fragments,
/// the stage reads and the addressing. It stays below the unit's limit.
const ACCUMULATOR_REGISTER_SHARE: usize = 4;

/// The counts every other candidate moves one setting off: two fragments along each axis on two
/// planes along each.
const BASE_FRAGMENTS: PlaneFragmentMicrokernelSettings =
    PlaneFragmentMicrokernelSettings::grid(2, 2, 1);
const BASE_PLANES: MatmulPlaneLayout = MatmulPlaneLayout::new(2, 2);

/// The counts of row fragments tried besides the base's.
const OTHER_ROW_FRAGMENTS: [usize; 2] = [1, 4];

/// The settings worth measuring on a device: a base setting and, around it, each setting moved off
/// it one at a time (the rows' fragments crossed with the ring, which share one budget; the
/// planes; a plane's fragments grown to the register budget down the rows; planes across the
/// output rows over the instruction of fewest rows). Interactions between those moves are not
/// listed. The device prunes the list; the problem only orders it.
pub struct ConvCandidates<'a> {
    hw: &'a Hardware,
    plane_width: usize,
}

impl<'a> ConvCandidates<'a> {
    /// The list for `hw`, or `None` where it commits to no plane width, which no matrix
    /// instruction runs without.
    pub fn on(hw: &'a Hardware) -> Option<Self> {
        let plane_width = hw.plane_dim().ok()? as usize;
        Some(Self { hw, plane_width })
    }

    /// Every candidate, the base first: the settings around the base of each instruction the
    /// device offers, then the base of each kind of instruction at one element a read.
    pub fn list(&self) -> Vec<ConvSettings> {
        let instructions: Vec<_> = [PlaneFragmentTransport::Cmma, PlaneFragmentTransport::Mma]
            .into_iter()
            .flat_map(|transport| {
                self.instruction_shapes(transport)
                    .into_iter()
                    .map(move |shape| (transport, shape))
            })
            .collect();
        let narrowest = instructions
            .iter()
            .enumerate()
            .min_by_key(|(_, (_, shape))| shape.rows)
            .map(|(position, _)| position);
        let mut list = Vec::new();
        let mut keep = |settings: ConvSettings, shape: &InstructionShape| {
            if self.fits(shape, &settings) && !list.contains(&settings) {
                list.push(settings);
            }
        };
        for (position, (transport, shape)) in instructions.iter().enumerate() {
            let base = Self::base(*transport, shape);
            let across = narrowest == Some(position);
            for settings in self.around(base, shape, across) {
                keep(settings, shape);
            }
            if instructions[..position]
                .iter()
                .all(|(other, _)| other != transport)
            {
                keep(base.width_one(), shape);
            }
        }
        list
    }

    /// The instructions of `transport` the device offers, one for each count of rows of each
    /// triple of elements, the one contracting the most first: a row count's columns and depth
    /// are the triple's own, not those of whichever element's shape came first.
    fn instruction_shapes(&self, transport: PlaneFragmentTransport) -> Vec<InstructionShape> {
        let configs: Vec<_> = match transport {
            PlaneFragmentTransport::Cmma => self
                .hw
                .properties
                .features
                .matmul
                .cmma
                .iter()
                .copied()
                .largest_first()
                .collect(),
            PlaneFragmentTransport::Mma => manual_mma(self.hw).largest_first().collect(),
            PlaneFragmentTransport::Wgmma => Vec::new(),
        };
        let mut seen: Vec<(ElemType, ElemType, ElemType, u32)> = Vec::new();
        let mut shapes = Vec::new();
        for config in configs {
            let key = (config.a_type, config.b_type, config.cd_type, config.m);
            if !seen.contains(&key) {
                seen.push(key);
                shapes.push(InstructionShape {
                    rows: config.m as usize,
                    columns: config.n as usize,
                    depth: config.k as usize,
                });
            }
        }
        shapes
    }

    /// The base setting of an instruction of `shape` reached through `transport`.
    fn base(transport: PlaneFragmentTransport, shape: &InstructionShape) -> ConvSettings {
        let base = ConvSettings::new(shape.rows, BASE_FRAGMENTS, BASE_PLANES);
        ConvSettings {
            microkernel: base.microkernel.through(transport),
            ..base
        }
    }

    /// The settings around `base`, an instruction of `shape`, the base first: fragments along the
    /// rows crossed with the ring up to the base's count, the planes cut, a plane grown to the
    /// register budget down the rows with its planes across the columns, and, where `across`, one
    /// instruction a plane with the planes spent across the output rows.
    fn around(
        &self,
        base: ConvSettings,
        shape: &InstructionShape,
        across: bool,
    ) -> Vec<ConvSettings> {
        let count = |count: usize| NonZeroUsize::new(count).expect("a count is not zero");
        let with =
            |plane: PlaneFragmentMicrokernelSettings, planes: MatmulPlaneLayout| ConvSettings {
                microkernel: CubeStagedMicrokernelSettings {
                    plane,
                    planes,
                    ..base.microkernel
                },
                ..base
            };
        let fragments = base.microkernel.plane;
        let planes = base.microkernel.planes;
        let with_rows = |rows: usize| {
            let plane = PlaneFragmentMicrokernelSettings {
                along_m: count(rows),
                ..fragments
            };
            with(plane, planes)
        };
        let base_rows = fragments.along_m.get();
        let mut around = Vec::new();
        for rows in [base_rows].into_iter().chain(OTHER_ROW_FRAGMENTS) {
            around.push(with_rows(rows));
            if rows <= base_rows {
                around.push(with_rows(rows).double_ring());
            }
        }
        for planes in [MatmulPlaneLayout::new(1, 2), MatmulPlaneLayout::new(1, 1)] {
            around.push(with(fragments, planes));
        }
        let grown = PlaneFragmentMicrokernelSettings {
            along_m: count(self.fragments_in_budget(shape)),
            along_n: NonZeroUsize::MIN,
            ..fragments
        };
        around.push(with(grown, MatmulPlaneLayout::new(1, 2)));
        if across {
            let one = PlaneFragmentMicrokernelSettings {
                along_m: NonZeroUsize::MIN,
                along_n: NonZeroUsize::MIN,
                ..fragments
            };
            let alone = with(one, MatmulPlaneLayout::new(1, 1));
            for planes in Self::counts(self.planes_across()) {
                around.push(alone.across(planes));
            }
        }
        around
    }

    /// Registers of a unit the accumulators of its plane's fragments may take. No runtime reports
    /// a unit's register file, so this assumes the one [`UNIT_REGISTERS`] names for every device.
    fn register_budget(&self) -> usize {
        UNIT_REGISTERS as usize / ACCUMULATOR_REGISTER_SHARE
    }

    /// Registers one fragment of `shape` takes in each unit of a plane.
    fn fragment_registers(&self, shape: &InstructionShape) -> usize {
        (shape.rows * shape.columns / self.plane_width.max(1)).max(1)
    }

    /// The most fragments of `shape` a plane holds along the rows with one along the columns,
    /// within the register budget.
    fn fragments_in_budget(&self, shape: &InstructionShape) -> usize {
        (self.register_budget() / self.fragment_registers(shape)).max(1)
    }

    /// The planes a cube holds along the across axis at most: the device's count of planes for a
    /// cube of its full units.
    fn planes_across(&self) -> usize {
        let hardware = &self.hw.properties.hardware;
        self.hw.plane_count_per_cube(
            hardware.max_units_per_cube as usize,
            self.plane_width as u32,
        ) as usize
    }

    /// The counts from two to `most` worth trying: the powers of two, and every count in the top
    /// octave, since the extents a map is cut into are rarely powers of two.
    fn counts(most: usize) -> impl Iterator<Item = usize> {
        (2..=most).filter(move |count| count.is_power_of_two() || *count > most / 2)
    }

    /// The bytes of the widest operand element a stage holds. The list is the device's, before any
    /// problem names its elements, so it assumes none is wider than `f32`: a stage of wider
    /// elements that does not fit is refused when it is prepared.
    fn widest_element_bytes() -> usize {
        ElemType::Float(FloatKind::F32).size()
    }

    /// Whether the device holds `settings` of `shape`: its plane count in a cube, its accumulators
    /// in a unit's registers, and one step of its stages, at the widest element, in shared memory.
    fn fits(&self, shape: &InstructionShape, settings: &ConvSettings) -> bool {
        let (fragments, planes) = (&settings.microkernel.plane, &settings.microkernel.planes);
        let (along_m, along_n) = (fragments.along_m.get(), fragments.along_n.get());
        let registers = along_m * along_n * self.fragment_registers(shape);
        let rows = shape.rows * along_m * planes.along_m.get();
        let columns = shape.columns * along_n * planes.along_n.get();
        let positions = rows * settings.across.get();
        let step = settings
            .microkernel
            .ring
            .bytes((positions + columns) * shape.depth * Self::widest_element_bytes());
        registers <= self.register_budget()
            && settings.planes() <= self.hw.full_cube_planes()
            && step <= self.hw.shared_memory_budget()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tiled::routines::test_support::{cmma_only, nvidia};
    use cubek_std::devices::topology;
    use cubek_std::microkernel::StageRing;

    fn list(hw: &Hardware) -> Vec<ConvSettings> {
        ConvCandidates::on(hw)
            .expect("the device commits to a plane width")
            .list()
    }

    fn most_across(hw: &Hardware) -> usize {
        list(hw)
            .iter()
            .map(|settings| settings.across.get())
            .max()
            .expect("the device lists settings")
    }

    /// The base is the first of the list: the largest instruction over 2x2 fragments on 2x2 planes.
    #[test]
    fn the_base_comes_first() {
        let base = list(&nvidia())[0];
        let microkernel = base.microkernel;
        assert_eq!(microkernel.plane.transport, PlaneFragmentTransport::Cmma);
        assert_eq!(
            microkernel.plane,
            PlaneFragmentMicrokernelSettings::grid(2, 2, 1)
        );
        assert_eq!(microkernel.planes, MatmulPlaneLayout::new(2, 2));
        assert_eq!(base.across, NonZeroUsize::MIN);
        assert_eq!(microkernel.ring, StageRing::SINGLE);
    }

    /// No setting is listed twice.
    #[test]
    fn no_setting_is_listed_twice() {
        let list = list(&nvidia());
        for (position, settings) in list.iter().enumerate() {
            assert!(!list[..position].contains(settings), "{settings}");
        }
    }

    /// A device lists the instructions it offers and no others: without the 8x32 no setting takes
    /// eight rows, and without a manual mma none takes it.
    #[test]
    fn a_device_lists_only_the_instructions_it_offers() {
        let eight = NonZeroUsize::new(8).unwrap();
        assert!(list(&nvidia()).iter().any(|s| s.rows == eight));
        let list = list(&cmma_only());
        assert!(!list.is_empty());
        for settings in list {
            assert_ne!(settings.rows, eight, "{settings}");
            assert_eq!(
                settings.microkernel.plane.transport,
                PlaneFragmentTransport::Cmma
            );
        }
    }

    /// A cube of fewer planes drops the large across splits, and a full one lists the splits
    /// that cut a map of seven rows and of eight.
    #[test]
    fn the_across_splits_follow_the_planes_of_a_cube() {
        let mut small = nvidia();
        topology(&mut small).max_units_per_cube = 128;
        assert!(most_across(&small) <= 4);
        assert!(most_across(&nvidia()) >= 7);
        let across: Vec<_> = list(&nvidia())
            .iter()
            .map(|settings| settings.across.get())
            .collect();
        assert!(across.contains(&7) && across.contains(&8));
    }

    /// Moving one setting at a time off the base keeps the list short.
    #[test]
    fn the_list_stays_short() {
        let len = list(&nvidia()).len();
        assert!((15..=30).contains(&len), "{len} settings");
    }

    /// Every setting's planes fit in a cube of the device.
    #[test]
    fn every_setting_fits_the_cube() {
        let hw = nvidia();
        for settings in list(&hw) {
            assert!(settings.planes() <= hw.full_cube_planes(), "{settings}");
        }
    }
}
