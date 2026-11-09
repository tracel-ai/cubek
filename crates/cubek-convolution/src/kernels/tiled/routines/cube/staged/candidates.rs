use crate::kernels::tiled::{
    ConvAlong, ConvGrid, ConvInstruction, ConvLines, ConvSettings, ConvWalk,
};
use core::num::NonZeroUsize;
use cubek_microkernel::matmul::plane::PlaneFragmentTransport;
use cubek_std::microkernel::StageRing;
use cubek_std::{Hardware, Instructions, manual_mma};

/// Accumulator registers one unit holds over the fragments of its plane before they spill.
const ACCUMULATOR_REGISTERS: usize = 64;

/// The share of a full cube's planes a cube spends on across positions, so a worker holds several
/// cubes and one waiting on a barrier does not idle it.
const ACROSS_SHARE: usize = 4;

/// Bytes of the widest operand element a stage holds, which the shared memory prune assumes.
const WIDEST_ELEMENT_BYTES: usize = 4;

/// One matrix instruction of the device, as the list reads it.
struct Tile {
    rows: usize,
    columns: usize,
    depth: usize,
}

/// How a count along one axis splits between the instructions of a plane and the planes of a cube.
struct Cut {
    fragments: usize,
    planes: usize,
}

/// The settings worth measuring on a device: a lead grid, crossed over how the rows and the columns
/// split into fragments and planes, with the across split, the ring and the width moved off it one
/// at a time. The device prunes the list; the problem only orders it.
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

    /// Every candidate, the lead first: for each instruction of each kind the device offers, the
    /// grids it holds at both rings, then the lead at one element a read.
    pub fn list(&self) -> Vec<ConvSettings> {
        let mut list = Vec::new();
        for transport in [PlaneFragmentTransport::Cmma, PlaneFragmentTransport::Mma] {
            for tile in self.tiles(transport) {
                let grids = self.grids(&tile);
                for &grid in &grids {
                    for ring in [StageRing::SINGLE, StageRing::DOUBLE] {
                        if self.fits(&tile, &grid, ring) {
                            list.push(Self::settings(transport, &tile, grid, ring));
                        }
                    }
                }
                if let Some(&lead) = grids.first() {
                    list.push(ConvSettings {
                        lines: ConvLines::One,
                        ..Self::settings(transport, &tile, lead, StageRing::SINGLE)
                    });
                }
            }
        }
        list
    }

    /// The instructions of `transport` the device offers, one for each count of rows, the one
    /// contracting the most first.
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
        let mut tiles: Vec<Tile> = Vec::new();
        for shape in shapes {
            let rows = shape.m as usize;
            if tiles.iter().all(|tile| tile.rows != rows) {
                tiles.push(Tile {
                    rows,
                    columns: shape.n as usize,
                    depth: shape.k as usize,
                });
            }
        }
        tiles
    }

    /// The grids around the lead: every split of the rows and of the columns, then planes across
    /// over one instruction a plane.
    fn grids(&self, tile: &Tile) -> Vec<ConvGrid> {
        let grid = |rows: &Cut, columns: &Cut, across: usize| ConvGrid {
            fragments: ConvAlong::new(rows.fragments, columns.fragments),
            planes: ConvAlong::new(rows.planes, columns.planes),
            across: NonZeroUsize::new(across).expect("a count across is at least one"),
        };
        let cut = |fragments, planes| Cut { fragments, planes };
        let rows = [cut(2, 2), cut(2, 1), cut(1, 2), cut(1, 1)]
            .into_iter()
            .chain((3..=ACCUMULATOR_REGISTERS).map(|fragments| cut(fragments, 1)));
        let columns = [cut(2, 2), cut(1, 2), cut(1, 1)];
        let mut grids = Vec::new();
        for rows in rows {
            for columns in &columns {
                grids.push(grid(&rows, columns, 1));
            }
        }
        let single = cut(1, 1);
        for across in 2..=self.hw.full_cube_planes() / ACROSS_SHARE {
            grids.push(grid(&single, &single, across));
        }
        grids.retain(|grid| self.fits(tile, grid, StageRing::SINGLE));
        grids
    }

    /// Whether the device holds `grid` of `tile`: its plane count in a cube, its accumulators in a
    /// unit's registers, and one step of its stages, at the widest element, in shared memory.
    fn fits(&self, tile: &Tile, grid: &ConvGrid, ring: StageRing) -> bool {
        let fragments = grid.fragments.m.get() * grid.fragments.n.get();
        let registers = fragments * tile.rows * tile.columns / self.plane_width.max(1);
        let planes = grid.planes.m.get() * grid.planes.n.get() * grid.across.get();
        let rows = tile.rows * grid.fragments.m.get() * grid.planes.m.get();
        let columns = tile.columns * grid.fragments.n.get() * grid.planes.n.get();
        let positions = rows * grid.across.get();
        let step = ring.bytes((positions + columns) * tile.depth * WIDEST_ELEMENT_BYTES);
        registers <= ACCUMULATOR_REGISTERS
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
    use crate::kernels::tiled::routines::test_support::{
        cmma_8x32_only, cmma_only, nvidia, with_plane_width,
    };

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

    /// A wider plane leaves a cube fewer planes, so it drops the large across splits.
    #[test]
    fn a_wide_plane_drops_large_across_splits() {
        let wide = most_across(&with_plane_width(nvidia(), 64));
        assert!(wide < most_across(&nvidia()));
        assert!(wide < 7);
        assert!(most_across(&cmma_8x32_only()) >= 7);
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
