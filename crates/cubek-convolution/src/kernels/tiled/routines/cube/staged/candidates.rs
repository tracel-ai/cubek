use crate::kernels::tiled::{
    ConvAlong, ConvGrid, ConvInstruction, ConvLines, ConvSettings, ConvWalk,
};
use core::num::NonZeroUsize;
use cubecl::ir::ElemType;
use cubek_microkernel::matmul::plane::PlaneFragmentTransport;
use cubek_std::microkernel::StageRing;
use cubek_std::{Hardware, Instructions, UNIT_REGISTERS, manual_mma};

/// The share of a unit's registers the accumulators take, the rest left to the operand fragments,
/// the stage reads and the addressing. An assumption kept below the unit's limit, not a measured
/// bound.
const ACCUMULATOR_REGISTER_SHARE: usize = 4;

/// Bytes of the widest operand element a stage holds, which the shared memory prune assumes.
const WIDEST_ELEMENT_BYTES: usize = 4;

/// The grid the others move one setting off: two fragments along each axis on two planes along each.
const LEAD: ConvGrid = ConvGrid {
    fragments: ConvAlong::new(2, 2),
    planes: ConvAlong::new(2, 2),
    across: NonZeroUsize::MIN,
};

/// The counts of row fragments tried, the lead's first.
const ROW_FRAGMENTS: [usize; 3] = [LEAD.fragments.m.get(), 1, 4];

/// One matrix instruction of the device, as the list reads it.
struct Tile {
    rows: usize,
    columns: usize,
    depth: usize,
}

/// The settings worth measuring on a device: a lead grid and, around it, each setting moved off it
/// one at a time (the rows' fragments crossed with the ring, which share one budget; the planes;
/// a plane's fragments grown to the register budget down the rows; planes across the output rows
/// over the instruction of fewest rows). Main effects only: interactions the star misses are the
/// ones it does not list. The device prunes the list; the problem only orders it.
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

    /// Every candidate, the lead first: the star of each instruction the device offers, then the
    /// lead of each kind of instruction at one element a read.
    pub fn list(&self) -> Vec<ConvSettings> {
        let tiles: Vec<_> = [PlaneFragmentTransport::Cmma, PlaneFragmentTransport::Mma]
            .into_iter()
            .flat_map(|transport| {
                self.tiles(transport)
                    .into_iter()
                    .map(move |tile| (transport, tile))
            })
            .collect();
        let narrowest = tiles
            .iter()
            .enumerate()
            .min_by_key(|(_, (_, tile))| tile.rows)
            .map(|(position, _)| position);
        let mut list = Vec::new();
        let mut keep = |settings: ConvSettings, tile: &Tile| {
            let fits = self.fits(tile, &settings.grid, settings.walk.ring);
            if fits && !list.contains(&settings) {
                list.push(settings);
            }
        };
        for (position, (transport, tile)) in tiles.iter().enumerate() {
            let across = narrowest == Some(position);
            for (grid, ring) in self.star(tile, across) {
                keep(Self::settings(*transport, tile, grid, ring), tile);
            }
            if tiles[..position].iter().all(|(other, _)| other != transport) {
                let lead = Self::settings(*transport, tile, LEAD, StageRing::SINGLE);
                keep(
                    ConvSettings {
                        lines: ConvLines::One,
                        ..lead
                    },
                    tile,
                );
            }
        }
        list
    }

    /// The instructions of `transport` the device offers, one for each count of rows of each
    /// triple of elements, the one contracting the most first: a row count's columns and depth
    /// are the triple's own, not those of whichever element's shape came first.
    fn tiles(&self, transport: PlaneFragmentTransport) -> Vec<Tile> {
        let shapes: Vec<_> = match transport {
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
        let mut tiles = Vec::new();
        for shape in shapes {
            let key = (shape.a_type, shape.b_type, shape.cd_type, shape.m);
            if !seen.contains(&key) {
                seen.push(key);
                tiles.push(Tile {
                    rows: shape.m as usize,
                    columns: shape.n as usize,
                    depth: shape.k as usize,
                });
            }
        }
        tiles
    }

    /// The grids of `tile` around the lead, each with its ring, the lead first: fragments along
    /// the rows crossed with the ring up to the lead's count, the planes cut, a plane grown to the
    /// register budget down the rows with its planes across the columns, and, where `across`,
    /// one instruction a plane with the planes spent across the output rows.
    fn star(&self, tile: &Tile, across: bool) -> Vec<(ConvGrid, StageRing)> {
        let with_rows = |rows: usize| ConvGrid {
            fragments: ConvAlong::new(rows, LEAD.fragments.n.get()),
            ..LEAD
        };
        let mut star = Vec::new();
        for rows in ROW_FRAGMENTS {
            star.push((with_rows(rows), StageRing::SINGLE));
            if rows <= LEAD.fragments.m.get() {
                star.push((with_rows(rows), StageRing::DOUBLE));
            }
        }
        for planes in [ConvAlong::new(1, 2), ConvAlong::new(1, 1)] {
            star.push((ConvGrid { planes, ..LEAD }, StageRing::SINGLE));
        }
        star.push((
            ConvGrid {
                fragments: ConvAlong::new(self.fragments_in_budget(tile), 1),
                planes: ConvAlong::new(1, 2),
                across: NonZeroUsize::MIN,
            },
            StageRing::SINGLE,
        ));
        if across {
            for planes in Self::counts(self.planes_across()) {
                star.push((
                    ConvGrid {
                        fragments: ConvAlong::new(1, 1),
                        planes: ConvAlong::new(1, 1),
                        across: NonZeroUsize::new(planes).expect("a count across is at least one"),
                    },
                    StageRing::SINGLE,
                ));
            }
        }
        star
    }

    /// Registers of a unit the accumulators of its plane's fragments may take.
    fn register_budget(&self) -> usize {
        UNIT_REGISTERS as usize / ACCUMULATOR_REGISTER_SHARE
    }

    /// Registers one fragment of `tile` takes in each unit of a plane.
    fn fragment_registers(&self, tile: &Tile) -> usize {
        (tile.rows * tile.columns / self.plane_width.max(1)).max(1)
    }

    /// The most fragments of `tile` a plane holds along the rows with one along the columns, within
    /// the register budget.
    fn fragments_in_budget(&self, tile: &Tile) -> usize {
        (self.register_budget() / self.fragment_registers(tile)).max(1)
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

    /// Whether the device holds `grid` of `tile`: its plane count in a cube, its accumulators in a
    /// unit's registers, and one step of its stages, at the widest element, in shared memory.
    fn fits(&self, tile: &Tile, grid: &ConvGrid, ring: StageRing) -> bool {
        let fragments = grid.fragments.m.get() * grid.fragments.n.get();
        let registers = fragments * self.fragment_registers(tile);
        let planes = grid.planes.m.get() * grid.planes.n.get() * grid.across.get();
        let rows = tile.rows * grid.fragments.m.get() * grid.planes.m.get();
        let columns = tile.columns * grid.fragments.n.get() * grid.planes.n.get();
        let positions = rows * grid.across.get();
        let step = ring.bytes((positions + columns) * tile.depth * WIDEST_ELEMENT_BYTES);
        registers <= self.register_budget()
            && planes <= self.hw.full_cube_planes()
            && step <= self.hw.shared_memory_budget()
    }

    fn settings(
        transport: PlaneFragmentTransport,
        tile: &Tile,
        grid: ConvGrid,
        ring: StageRing,
    ) -> ConvSettings {
        ConvSettings {
            instruction: ConvInstruction {
                transport,
                rows: NonZeroUsize::new(tile.rows).expect("an instruction has rows"),
            },
            grid,
            walk: ConvWalk {
                ring,
                ..ConvWalk::DEFAULT
            },
            lines: ConvLines::Widest,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::tiled::CubeStagedConvRoutine;
    use crate::kernels::tiled::routines::test_support::{cmma_only, nvidia};
    use cubek_std::devices::topology;

    fn list(hw: &Hardware) -> Vec<ConvSettings> {
        ConvCandidates::on(hw)
            .expect("the device commits to a plane width")
            .list()
    }

    fn most_across(hw: &Hardware) -> usize {
        list(hw)
            .iter()
            .map(|settings| settings.grid.across.get())
            .max()
            .expect("the device lists settings")
    }

    /// The lead is the first of the list: the largest instruction over 2x2 fragments on 2x2 planes.
    #[test]
    fn the_lead_comes_first() {
        let lead = list(&nvidia())[0];
        assert_eq!(lead.instruction.transport, PlaneFragmentTransport::Cmma);
        assert_eq!(lead.grid.fragments, ConvAlong::new(2, 2));
        assert_eq!(lead.grid.planes, ConvAlong::new(2, 2));
        assert_eq!(lead.grid.across, NonZeroUsize::MIN);
        assert_eq!(lead.walk.ring, StageRing::SINGLE);
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
        assert!(list(&nvidia()).iter().any(|s| s.instruction.rows == eight));
        let list = list(&cmma_only());
        assert!(!list.is_empty());
        for settings in list {
            assert_ne!(settings.instruction.rows, eight, "{settings}");
            assert_eq!(settings.instruction.transport, PlaneFragmentTransport::Cmma);
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
            .map(|settings| settings.grid.across.get())
            .collect();
        assert!(across.contains(&7) && across.contains(&8));
    }

    /// A star keeps the list short.
    #[test]
    fn the_list_is_a_star() {
        let len = list(&nvidia()).len();
        assert!((15..=30).contains(&len), "{len} settings");
    }

    /// The settings that win measured problems are on the list, derived from the instructions
    /// the device offers: a plane of seven 8-row fragments, one 8-row plane for each of seven
    /// rows, two 16-row fragments on planes across the columns, and Mma at two stages.
    #[test]
    fn the_settings_that_win_are_listed() {
        let list = list(&nvidia());
        for settings in [
            CubeStagedConvRoutine::F7X1_P1X2_T8,
            CubeStagedConvRoutine::F1X1_P1X1_T8_A7,
            CubeStagedConvRoutine::F2X2_P1X2,
            CubeStagedConvRoutine::F1X2_P2X2.mma().double_ring(),
        ] {
            assert!(list.contains(&settings), "{settings}");
        }
    }

    /// Every setting holds in a cube of the device: its planes, and the accumulators of its units.
    #[test]
    fn every_setting_fits_the_cube() {
        let hw = nvidia();
        for settings in list(&hw) {
            let planes = settings.grid.planes.m.get()
                * settings.grid.planes.n.get()
                * settings.grid.across.get();
            assert!(planes <= hw.full_cube_planes(), "{settings}");
        }
    }
}
